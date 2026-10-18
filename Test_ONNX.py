import onnxruntime as ort
import numpy as np
from PIL import Image

session=ort.InferenceSession("./onnx_data/lenet.onnx")

for inp in session.get_inputs():
	print(f"输入名：{inp.name}，形状：{inp.shape}，类型：{inp.type}")
for out in session.get_outputs():
	print(f"输出名：{out.name}，形状：{out.shape}")

input_name=session.get_inputs()[0].name
output_name=session.get_outputs()[0].name

for index in range(10):
	img=Image.open(f"./Img_data/test_{index}.png").convert("L")
	print(f"图像原始尺寸：{img.size}")

	img =img.resize((28,28))
	img_array=np.array(img).astype(np.float32)
	img_array=img_array/255.0
	img_array=(img_array-0.1307)/0.3801

	input_data=img_array[np.newaxis,np.newaxis,:,:]

	outputs=session.run([output_name],{input_name:input_data})
	logits=outputs[0]

	pred_class=np.argmax(logits,axis=1)[0]
	prob=np.exp(logits)/np.sum(np.exp(logits),axis=1,keepdims=True)
	confidence=prob[0][pred_class]

	print(f"预测数字：{pred_class}，置信度：{confidence:.4f}")
	print(f"各类别概率：{prob[0]}")

