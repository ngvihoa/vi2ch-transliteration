---
license: mit
language:
- vi
---
```python
import torch
from transformers import AutoModel, AutoTokenizer

model_path = 'CjangCjengh/NomBert-hn2qn-v0.1'
device = 'cuda'

model = AutoModel.from_pretrained(model_path, torch_dtype='auto', trust_remote_code=True).eval().to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

with torch.inference_mode():
    output_text, output_probs = model.parse_nom_text(tokenizer, ['仍調𬖉𧡊㐌𤴬疸𢚸'])
    print(output_text[0])
    # những điều trông thấy đã đau đớn lòng
    print(output_probs[0])
    # [
    # {'char': '仍', 'candidates': [('những', 0.5237383842468262), ('nhưng', 0.475042462348938), ('dưng', 0.0008663760963827372), ('nhang', 0.00022805406479164958), ('dừng', 8.42325171106495e-05), ('nhẵng', 1.6380783563363366e-05), ('nhùng', 1.5950208762660623e-05), ('nhửng', 3.0440487535088323e-06), ('nhăng', 2.9528700906666927e-06), ('nhẳng', 1.0688020211091498e-06), ('nhừng', 5.84112399337755e-07), ('nhâng', 5.119333650327462e-07)]},
    # {'char': '調', 'candidates': [('điều', 0.8831620812416077), ('đều', 0.11558306217193604), ('điệu', 0.0012446790933609009), ('dìu', 8.889981472748332e-06), ('điu', 7.615183221787447e-07), ('đìu', 5.942594043517602e-07)]},
    # {'char': '𬖉', 'candidates': [('trông', 1.0)]},
    # {'char': '𧡊', 'candidates': [('thấy', 1.0)]},
    # {'char': '㐌', 'candidates': [('đã', 0.9998464584350586), ('dã', 0.00014108473260421306), ('đà', 1.2395633348205592e-05)]},
    # {'char': '𤴬', 'candidates': [('đau', 0.9999825954437256), ('đáu', 1.744620021781884e-05)]},
    # {'char': '疸', 'candidates': [('đớn', 0.9998302459716797), ('đơn', 0.00014517175441142172), ('đảm', 2.457975824654568e-05)]},
    # {'char': '𢚸', 'candidates': [('lòng', 1.0)]}
    # ]
```