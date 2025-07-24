import base64
from typing import Optional, List
import cv2
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

ZXING_AVAILABLE = False
try:
    import zxingcpp
    ZXING_AVAILABLE = True
except Exception:
    pass

PYZBAR_AVAILABLE = False
try:
    from pyzbar.pyzbar import decode as pyzbar_decode, ZBarSymbol
    PYZBAR_AVAILABLE = True
except Exception:
    pass

app = FastAPI(title="Barcode Decoder API with All Codes & Debug", version="1.0.0")


class DecodeRequest(BaseModel):
    image_base64: str = Field(..., description="Base64字符串，可带data:image/...;base64,前缀")


class SimpleResponse(BaseModel):
    data: Optional[str] = None


def base64_to_cv2(b64: str) -> np.ndarray:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    img_bytes = base64.b64decode(b64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法从base64解码成图像")
    return img


def preprocess_strong(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.medianBlur(gray, 3)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def rotate_bound(image: np.ndarray, angle: float) -> np.ndarray:
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    return cv2.warpAffine(image, M, (nW, nH))


def decode_with_zxing(img_bgr: np.ndarray) -> Optional[str]:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = zxingcpp.read_barcodes(img_rgb)
    if results:
        # 不筛选格式，返回第一个识别结果
        return results[0].text
    return None


def decode_with_pyzbar(img_bgr: np.ndarray) -> Optional[str]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    results = pyzbar_decode(gray)  # 不限制格式，识别所有码
    if results:
        return results[0].data.decode("utf-8", "ignore")
    return None


def try_decode(img_bgr: np.ndarray) -> Optional[str]:
    if ZXING_AVAILABLE:
        try:
            r = decode_with_zxing(img_bgr)
            if r:
                return r
        except Exception:
            pass
    if PYZBAR_AVAILABLE:
        try:
            r = decode_with_pyzbar(img_bgr)
            if r:
                return r
        except Exception:
            pass
    return None


def try_decode_with_rotations(img_bgr: np.ndarray) -> Optional[str]:
    angles = [0, 90, 180, 270]
    imgs_to_try = []
    for angle in angles:
        if angle == 0:
            imgs_to_try.append(img_bgr)
        else:
            imgs_to_try.append(rotate_bound(img_bgr, angle))

    for img in imgs_to_try:
        pre_img = preprocess_strong(img)
        res = try_decode(pre_img)
        if res:
            return res
        res2 = try_decode(img)
        if res2:
            return res2
    return None


def locate_barcode_areas(img_bgr: np.ndarray, max_candidates=5, min_area_ratio=0.01, max_area_ratio=0.5) -> List[np.ndarray]:
    """
    定位图像中可能包含条形码的区域
    
    参数:
        img_bgr: 输入的BGR格式图像
        max_candidates: 返回的最大候选区域数量
        min_area_ratio: 候选区域最小面积与图像面积的比值
        max_area_ratio: 候选区域最大面积与图像面积的比值
    
    返回:
        候选区域图像列表, 候选区域边界信息列表(包含坐标和旋转矩形框)
    """
    height, width = img_bgr.shape[:2]
    min_area = min_area_ratio * height * width
    max_area = max_area_ratio * height * width
    
    # 转换为灰度图并进行模糊处理
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 计算水平方向梯度，条形码通常在水平方向有明显边缘变化
    grad_x = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad_y = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    
    # 计算梯度幅值并归一化
    gradient = cv2.subtract(grad_x, grad_y)
    gradient = cv2.convertScaleAbs(gradient)
    
    # 应用自适应阈值处理
    blurred = cv2.blur(gradient, (9, 9))
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 优化的形态学操作序列
    kernel_size = (int(width * 0.03), int(height * 0.01))  # 动态调整核大小
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    # 闭运算连接条形码区域
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 开运算去除小噪点
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 膨胀操作扩大条形码区域
    closed = cv2.dilate(closed, None, iterations=3)
    closed = cv2.erode(closed, None, iterations=2)
    
    # 查找轮廓并筛选
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 根据面积和长宽比筛选候选区域
    candidates = []
    return_candidates = []
    
    # 计算图像的宽高比
    image_aspect_ratio = width / height
    
    for c in contours:
        area = cv2.contourArea(c)
        
        # 过滤面积不符合要求的轮廓
        if area < min_area or area > max_area:
            continue
            
        # 计算轮廓的最小外接矩形
        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), angle = rect
        
        # 计算矩形的宽高比
        box_aspect_ratio = max(rw, rh) / min(rw, rh) if min(rw, rh) > 0 else 0
        
        # 条形码通常具有较高的宽高比
        if box_aspect_ratio < 2.0:  # 可以调整此阈值
            continue
            
        # 计算旋转矩形的四个顶点
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # 计算边界框坐标并确保在图像范围内
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x1 = max(min(xs), 0)
        x2 = min(max(xs), width)
        y1 = max(min(ys), 0)
        y2 = min(max(ys), height)
        
        # 确保边界框尺寸合理
        if x2 - x1 < 10 or y2 - y1 < 10:
            continue
            
        # 添加候选区域
        candidates.append(img_bgr[y1:y2, x1:x2])
        return_candidates.append((x1, y1, x2, y2, box))
    
    # 根据面积排序，保留最大的几个候选区域
    sorted_indices = sorted(range(len(candidates)), 
                           key=lambda i: (return_candidates[i][2] - return_candidates[i][0]) * 
                                         (return_candidates[i][3] - return_candidates[i][1]), 
                           reverse=True)
    
    candidates = [candidates[i] for i in sorted_indices[:max_candidates]]
    return_candidates = [return_candidates[i] for i in sorted_indices[:max_candidates]]
    
    return candidates, return_candidates


@app.post("/decode", response_model=SimpleResponse)
def decode_endpoint(req: DecodeRequest):
    try:
        img = base64_to_cv2(req.image_base64)
        # candidates, boxes = locate_barcode_areas(img)
        # # 画框选调试图
        # img_debug = img.copy()
        # for (x1, y1, x2, y2, box) in boxes:
        #     cv2.polylines(img_debug, [box], isClosed=True, color=(0, 255, 0), thickness=2)
        # cv2.imwrite("1.png", img_debug)

        # for area in candidates:
        #     res = try_decode_with_rotations(area)
        #     if res:
        #         return SimpleResponse(data=res)

        # 整图再尝试识别
        result = try_decode_with_rotations(img)
        return SimpleResponse(data=result)
    except Exception as e:
        print("识别异常:", e)
        return SimpleResponse(data=None)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
