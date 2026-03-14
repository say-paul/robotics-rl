run vnc sever + simulation

podman run -d --name g1-vnc --replace --network=host g1-vnc
DISPLAY=:99 python vnc/sim_viewer.py --policy policies/g1_stand_v5.onnx

