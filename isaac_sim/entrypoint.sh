#!/bin/bash
# This project was developed with assistance from AI tools.
set -e

export CYCLONEDDS_HOME=${CYCLONEDDS_HOME:-/opt/cyclonedds}
# Override container's default streaming mode — prevents sim_app timeout
unset LIVESTREAM
export HEADLESS=1

# Enable multicast on loopback for DDS
ip link set lo multicast on 2>/dev/null || true

# Patch Unitree SDK DDS config: enable multicast for all configs
find /opt/unitree_sim /isaac-sim/kit -path "*/unitree_sdk2py/core/channel_config.py" 2>/dev/null | while read f; do
    /isaac-sim/kit/python/bin/python3 -c "
with open('$f') as fh: c = fh.read()
c = c.replace('multicast=\"default\"', 'multicast=\"true\"')
c = c.replace('multicast=\\\\\"default\\\\\"', 'multicast=\\\\\"true\\\\\"')
c = c.replace('multicast=\"false\"', 'multicast=\"true\"')
with open('$f', 'w') as fh: fh.write(c)
print(f'Patched: $f')
"
done
# Restore Domain() call (our earlier patch broke cross-process discovery)
find /opt/unitree_sim /isaac-sim/kit -path "*/unitree_sdk2py/core/channel.py" 2>/dev/null | while read f; do
    sed -i 's/pass  # patched: skip Domain, use DomainParticipant only/self.__domain = Domain(id, config)/' "$f" 2>/dev/null
done

# Copy missing .so files for unitree_sdk2py CRC (from /groot if mounted, skip otherwise)
SDK_LIB="/isaac-sim/kit/python/lib/python3.11/site-packages/unitree_sdk2py/utils/lib"
if [ -d "/groot/external_dependencies/unitree_sdk2_python/unitree_sdk2py/utils/lib" ]; then
    mkdir -p "$SDK_LIB"
    cp -f /groot/external_dependencies/unitree_sdk2_python/unitree_sdk2py/utils/lib/*.so "$SDK_LIB/"
    echo "[entrypoint] Copied CRC .so from /groot"
elif [ ! -f "$SDK_LIB/crc_amd64.so" ]; then
    echo "[entrypoint] Warning: CRC .so not found. Mount GR00T-WBC at /groot or bake into image."
fi

# Write run_command shm name to a file the viewer sidecar can read
# (both containers share /dev/shm via the dshm volume)
# Clean stale shared memory from previous runs
rm -f /dev/shm/psm_* 2>/dev/null

# Upgrade camera resolution to 720p
sed -i 's/height: int = 480/height: int = 720/g; s/width: int =  640/width: int = 1280/g; s/height=480/height=720/g; s/width=640/width=1280/g' \
    /opt/unitree_sim/tasks/common_config/camera_configs.py 2>/dev/null

# Replace right_wrist_camera with world_camera in the shared memory writer
# so the teleimager publishes the third-person view on port 55557
sed -i 's/right_wrist_camera/world_camera/g' \
    /opt/unitree_sim/tasks/common_observations/camera_state.py 2>/dev/null

echo "[entrypoint] Starting G1 wholebody simulation..."
# The shm name will be written by a background watcher after DDS init
(while true; do
    SHM=$(grep -o 'psm_[a-f0-9]*' /proc/1/fd/1 2>/dev/null | tail -1)
    if [ -n "$SHM" ]; then
        # Find the run_command output shm name from logs
        RUN_CMD_SHM=$(grep 'run_command_dds.*Output shared memory' /proc/1/fd/1 2>/dev/null | grep -o 'psm_[a-f0-9]*')
        if [ -n "$RUN_CMD_SHM" ]; then
            echo "$RUN_CMD_SHM" > /dev/shm/run_command_shm_name
            break
        fi
    fi
    sleep 5
done) &
cd /opt/unitree_sim && exec /isaac-sim/python.sh sim_main.py \
    --task Isaac-Move-Cylinder-G129-Dex1-Wholebody \
    --action_source dds_wholebody \
    --robot_type g129 \
    --enable_dex1_dds \
    --enable_wholebody_dds \
    --headless \
    --enable_cameras \
    --camera_jpeg_quality 95 \
    --camera_include "front_camera,right_wrist_camera" \
    "$@"
