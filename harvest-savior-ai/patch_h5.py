import json
import h5py

path = 'C:/Users/Preetham Rao/Desktop/HarvestSavior/harvest-savior-ai/model/crop_disease_cnn.h5'
with h5py.File(path, 'r+') as f:
    c = json.loads(f.attrs['model_config'])
    
    # We will remove the scalar positional arguments
    # and instead rely on Keras handling it or we just patch the config so TrueDivide has no args[1]
    # But wait, if TrueDivide has no args[1], it will crash.
    # What if we change the class_name to "Lambda" and pass a function?
    # Keras 3 lambda serialization is complex.
    
    changed = False
    for layer in c['config']['layers']:
        if layer['class_name'] in ('TrueDivide', 'Subtract'):
            for node in layer['inbound_nodes']:
                args = node['args']
                if len(args) == 2 and isinstance(args[1], (int, float)):
                    val = args.pop(1)
                    # Use 'y' for mathematical layers
                    node['kwargs']['y'] = val
                    changed = True

    if changed:
        f.attrs['model_config'] = json.dumps(c).encode('utf-8')
        print('Patched H5 JSON successfully!')
    else:
        print('No changes needed.')
