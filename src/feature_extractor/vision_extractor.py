"""
视觉特征提取器
使用 ResNet18 提取帧级特征，再通过 PCA 降至 35 维

策略:
  1. 使用 ffmpeg 从视频抽帧
  2. ResNet18 提取每帧的 512 维特征
  3. 取前 35 维或 PCA 降到 35 维 (与附件2 vision 对齐)
  4. 对齐到 50 帧

注:
  由于 OpenFace 编译困难,采用预训练 ResNet18 作为视觉特征提取的替代方案。
  ResNet 特征虽不直接对应 OpenFace 的 AU/pose/gaze 维度,但语义信息更丰富,
  在多模态情感分析中已被验证有效(参考文献[3])。
"""
import numpy as np
import cv2
import os
import subprocess
import tempfile
from typing import Union, Optional
import torch
import torchvision.models as models
import torchvision.transforms as transforms


class VisionFeatureExtractor:
    """
    从视频文件提取视觉特征

    输出形状: (num_frames, 35) float32
    - 35 维特征 (与附件2 vision 维度一致)
    - 时间帧数: 默认 50 帧
    """

    def __init__(self,
                 device: str = None,
                 target_frames: int = 50,
                 feature_dim: int = 35,
                 model_name: str = 'resnet18',
                 img_size: int = 224):
        """
        Args:
            device: 'cuda'/'cpu', None 则自动选择
            target_frames: 目标时间帧数（与附件2 aligned_50 对齐）
            feature_dim: 输出特征维度（与附件2一致为35）
            model_name: 预训练模型名
            img_size: 输入图像尺寸
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.target_frames = target_frames
        self.feature_dim = feature_dim
        self.img_size = img_size

        print(f"[VisionExtractor] 加载模型: {model_name} -> {self.device}")
        # 加载预训练模型
        if model_name == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.model = models.resnet18(weights=weights)
            # 去掉最后分类层,保留 avg pool 之前的特征
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            raw_dim = 512
        elif model_name == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            self.model = models.resnet50(weights=weights)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            raw_dim = 2048
        else:
            raise ValueError(f"不支持的模型: {model_name}")

        self.model.eval()
        self.model.to(self.device)
        self.raw_dim = raw_dim

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])

        # 投影矩阵: 训练阶段会用 attachment2 数据训练,这里先用随机投影
        # 实际使用前应调用 fit_projection 训练
        self.projection = None  # shape: (raw_dim, feature_dim)

    def extract_frames_from_video(self, video_path: str,
                                   num_frames: int = None) -> list:
        """使用 ffmpeg 抽帧"""
        if num_frames is None:
            num_frames = self.target_frames

        # 获取视频时长
        import json
        probe_cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', video_path
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, check=True)
        duration = float(json.loads(probe_result.stdout)['format']['duration'])

        # 计算采样间隔
        fps = num_frames / duration if duration > 0 else 30.0

        # 创建临时目录
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 抽帧
            cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-vf', f'fps={fps}',
                '-q:v', '2',  # 图像质量
                os.path.join(tmp_dir, 'frame_%04d.jpg')
            ]
            subprocess.run(cmd, capture_output=True, check=True)

            # 读取所有帧
            frames = []
            for fname in sorted(os.listdir(tmp_dir)):
                if fname.endswith('.jpg'):
                    img = cv2.imread(os.path.join(tmp_dir, fname))
                    if img is not None:
                        frames.append(img)
        return frames

    def extract_features_from_frames(self, frames: list) -> np.ndarray:
        """从帧列表提取特征"""
        if len(frames) == 0:
            return np.zeros((self.target_frames, self.feature_dim), dtype=np.float32)

        # 预处理 + 批量推理
        feats = []
        batch_size = 16

        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batch_tensors = []
            for img in batch:
                # BGR -> RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tensor = self.transform(img_rgb)
                batch_tensors.append(tensor)

            batch_tensor = torch.stack(batch_tensors).to(self.device)
            with torch.no_grad():
                feat = self.model(batch_tensor)
            # feat shape: (B, raw_dim, 1, 1) -> squeeze
            feat = feat.squeeze(-1).squeeze(-1).cpu().numpy()  # (B, raw_dim)
            feats.append(feat)

        feats = np.concatenate(feats, axis=0)  # (T, raw_dim)

        # 对齐到 target_frames
        feats = self._align_to_target(feats, self.target_frames)

        # 投影到 35 维
        feats = self._project_to_dim(feats, self.feature_dim)

        return feats.astype(np.float32)

    def _align_to_target(self, feat: np.ndarray, target: int) -> np.ndarray:
        """对齐到目标帧数"""
        T = feat.shape[0]
        if T == target:
            return feat
        elif T > target:
            # 均匀采样
            indices = np.linspace(0, T - 1, target).astype(int)
            return feat[indices]
        else:
            # 重复插值
            repeat_factor = target // T + 1
            feat = np.repeat(feat, repeat_factor, axis=0)[:target]
            return feat

    def _project_to_dim(self, feat: np.ndarray, target_dim: int) -> np.ndarray:
        """投影到目标维度"""
        if self.projection is not None:
            return feat @ self.projection
        elif feat.shape[1] > target_dim:
            # 简单截断 (实际应用中应训练 PCA 或 NN 投影)
            return feat[:, :target_dim]
        else:
            # 零填充
            pad = np.zeros((feat.shape[0], target_dim - feat.shape[1]))
            return np.concatenate([feat, pad], axis=1)

    def fit_projection(self, frames_list: list, target_features: np.ndarray):
        """
        用附件2的 vision 特征训练投影矩阵

        Args:
            frames_list: 多个视频抽出的帧列表, shape [(T_i, raw_dim)]
            target_features: 附件2 的 vision 特征, shape (N, T, 35)
        """
        # 简单使用线性回归拟合
        # 更严谨的: PCA -> Linear
        from sklearn.decomposition import PCA
        from sklearn.linear_model import Ridge

        # 拼接所有帧特征
        all_feats = np.concatenate([f for f in frames_list if len(f) > 0], axis=0)  # (Sum_T, raw_dim)

        # 目标特征对齐到帧
        all_targets = []
        for i, feat in enumerate(frames_list):
            if len(feat) > 0:
                aligned_target = self._align_to_target(target_features[i], feat.shape[0])
                all_targets.append(aligned_target)
        all_targets = np.concatenate(all_targets, axis=0)  # (Sum_T, 35)

        # PCA 降维到中间维度
        pca = PCA(n_components=min(64, all_feats.shape[1]))
        pca_feats = pca.fit_transform(all_feats)

        # Ridge 回归拟合
        ridge = Ridge(alpha=1.0)
        ridge.fit(pca_feats, all_targets)

        # 存储投影
        self.projection = pca.components_.T @ ridge.coef_.T  # (raw_dim, 35)
        self.projection = self.projection.astype(np.float32)

        print(f"[VisionExtractor] 投影矩阵训练完成, shape: {self.projection.shape}")
        return self.projection

    def extract_for_sample(self, video_path: str) -> dict:
        """
        提取单条样本的视觉特征
        """
        frames = self.extract_frames_from_video(video_path)
        feat = self.extract_features_from_frames(frames)  # (50, 35)
        return {
            'vision': feat,  # (50, 35)
        }
