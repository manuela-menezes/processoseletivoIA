import tensorflow as tf
import os
 
#  1. Carregar o modelo treinado 
model = tf.keras.models.load_model("model.h5")
print("Modelo carregado de model.h5")
 
#  2. Técnica A: Dynamic Range Quantization (baseline) 
# Quantiza apenas os PESOS de float32 -> int8.
# Ativações permanecem em float32 durante a inferência.
# Vantagem: não requer dataset de calibração -- conversão simples e rápida.
# Limitação: sem quantização das ativações, conversões float<->int ocorrem
# em runtime, o que aumenta latência em hardware embarcado.
 
converter_drq = tf.lite.TFLiteConverter.from_keras_model(model)
converter_drq.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_drq = converter_drq.convert()
 
drq_kb = len(tflite_drq) / 1024
print(f"\nTécnica A — Dynamic Range Quantization")
print(f"  Tamanho em memória : {drq_kb:.1f} KB")
print(f"  Ativações em int8  : Não")
print(f"  Calibração         : Não necessária")
 
#  3. Técnica B: Full Integer Quantization (técnica principal) 
# Quantiza PESOS e ATIVAÇÕES em int8 com dataset de calibração.
# O conversor mede os ranges reais de cada tensor para definir
# fatores de escala (scale) e offset (zero-point) por camada.
# Resultado: inferência sem conversões float<->int em runtime --
# menor latência e menor consumo energético em hardware com
# aceleração inteira (ARM Cortex-M, ESP32-S3).
 
(x_cal, _), _ = tf.keras.datasets.mnist.load_data()
x_cal = x_cal[:100].astype("float32") / 255.0
x_cal = x_cal[..., tf.newaxis]  # (100, 28, 28, 1)
 
def representative_dataset():
    for img in x_cal:
        yield [img[tf.newaxis, ...]]  # (1, 28, 28, 1)
 
converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_dataset
converter_int8.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS,  # fallback para ops sem suporte nativo int8
]
tflite_int8 = converter_int8.convert()
 
int8_kb = len(tflite_int8) / 1024
print(f"\nTécnica B -- Full Integer Quantization")
print(f"  Tamanho em memória : {int8_kb:.1f} KB")
print(f"  Ativações em int8  : Sim")
print(f"  Calibração         : 100 amostras MNIST")
 
#  4. Salvar apenas o modelo final (Full Int8) 
# O Full Int8 é escolhido por ser mais adequado para deploy em Edge AI real.
 
with open("model.tflite", "wb") as f:
    f.write(tflite_int8)
 
print("\nModelo otimizado salvo em model.tflite")
 
#  5. Comparativo final 
h5_kb = os.path.getsize("model.h5") / 1024
 
print(f"\n--- Comparativo de tamanho ---")
print(f"{'Formato':<30} {'Tamanho':>10}  {'Redução vs .h5':>15}")
print(f"{'-'*57}")
print(f"{'model.h5 (referência)':<30} {h5_kb:>9.1f}KB  {'—':>15}")
print(f"{'Dynamic Range (não salvo)':<30} {drq_kb:>9.1f}KB  {(1-drq_kb/h5_kb)*100:>14.1f}%")
print(f"{'model.tflite (Full Int8)':<30} {int8_kb:>9.1f}KB  {(1-int8_kb/h5_kb)*100:>14.1f}%")
print(f"\nEscolha: Full Int8 -- ativações em int8 eliminam conversões em runtime,")
print(f"reduzindo latência e consumo energético em hardware embarcado.")
 

