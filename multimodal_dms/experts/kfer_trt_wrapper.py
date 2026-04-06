import numpy as np
import torch
import logging

log = logging.getLogger(__name__)

KFER_LABELS = ["angry", "anxious", "happy", "hurt", "neutral", "sad", "surprised"]
MAX_ENTROPY = float(np.log(7))


def _make_default_au(img_size=224):
    s = img_size
    return np.array([
        [s*0.50, s*0.22], [s*0.35, s*0.38], [s*0.65, s*0.38], [s*0.50, s*0.52],
        [s*0.22, s*0.55], [s*0.78, s*0.55], [s*0.50, s*0.72], [s*0.50, s*0.88],
    ], dtype=np.float32)


class KFERTRTExpert:
    """K-FER TensorRT inference wrapper. Same interface as KFERExpert."""

    def __init__(self, engine_path, device="cuda"):
        import tensorrt as trt
        self.device = device
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self._engine = runtime.deserialize_cuda_engine(f.read())
        self._ctx = self._engine.create_execution_context()

        self._in_names = []
        self._out_names = []
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._in_names.append(name)
            else:
                self._out_names.append(name)

        self.default_au = _make_default_au()
        self.id2label = {i: l for i, l in enumerate(KFER_LABELS)}
        log.info("K-FER TRT engine loaded: %s (inputs=%s)", engine_path, self._in_names)

    @torch.no_grad()
    def extract(self, faces_np, au_coords=None):
        empty = {"probs": np.zeros(7, dtype=np.float32), "entropy": MAX_ENTROPY,
                 "quality": 0.0, "top1_id": -1, "top1_label": "unknown",
                 "top1_conf": 0.0, "valid": False}

        if faces_np is None or len(faces_np) == 0:
            return empty

        # Take first valid frame
        frame = faces_np[0] if faces_np.ndim == 4 else faces_np
        if np.linalg.norm(frame) < 1.0:
            return empty

        # Normalize (same as PyTorch path)
        img = torch.from_numpy(frame).unsqueeze(0).float().cuda() / 255.0
        # MobileViTv2 normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).cuda()
        std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).cuda()
        img = (img - mean) / std

        if au_coords is None:
            au_coords = self.default_au
        au = torch.from_numpy(au_coords).unsqueeze(0).float().cuda()

        # Set shapes
        self._ctx.set_input_shape(self._in_names[0], tuple(img.shape))
        if len(self._in_names) > 1:
            self._ctx.set_input_shape(self._in_names[1], tuple(au.shape))

        out_shape = tuple(self._ctx.get_tensor_shape(self._out_names[0]))
        out_tensor = torch.empty(out_shape, dtype=torch.float32, device="cuda")

        self._ctx.set_tensor_address(self._in_names[0], img.data_ptr())
        if len(self._in_names) > 1:
            self._ctx.set_tensor_address(self._in_names[1], au.data_ptr())
        self._ctx.set_tensor_address(self._out_names[0], out_tensor.data_ptr())

        self._ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()

        logits = out_tensor[0].cpu().numpy()
        probs = np.exp(logits) / np.exp(logits).sum()  # softmax
        top1 = int(np.argmax(probs))
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))

        return {
            "probs": probs.astype(np.float32),
            "entropy": entropy,
            "quality": max(0.0, 1.0 - entropy / MAX_ENTROPY),
            "top1_id": top1,
            "top1_label": self.id2label[top1],
            "top1_conf": float(probs[top1]),
            "valid": True,
        }
