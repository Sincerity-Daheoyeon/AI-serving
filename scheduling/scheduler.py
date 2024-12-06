from flask import app
import time
import AI.init
from model.model import Model

model = Model(
        input_shape_structured=(3,),
        input_shape_unstructured=(None, 128, 128, 1),
    )
# model.load_weights(
    #     os.path.join(config['WEIGHT']['PATH'], config['WEIGHT']['FILE']),
    # )
while True:
    try:
        AI.init.run_model(model)
    except Exception as e:
        print(e)
    time.sleep(2)