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

# Copy missing .so files for unitree_sdk2py CRC
SDK_LIB="/isaac-sim/kit/python/lib/python3.11/site-packages/unitree_sdk2py/utils/lib"
if [ -d "/groot/external_dependencies/unitree_sdk2_python/unitree_sdk2py/utils/lib" ]; then
    mkdir -p "$SDK_LIB"
    cp -f /groot/external_dependencies/unitree_sdk2_python/unitree_sdk2py/utils/lib/*.so "$SDK_LIB/"
    echo "[entrypoint] Copied CRC .so to $SDK_LIB/"
    ls "$SDK_LIB/"
fi

echo "[entrypoint] Starting G1 wholebody simulation..."
cd /opt/unitree_sim && exec /isaac-sim/python.sh sim_main.py \
    --task Isaac-Move-Cylinder-G129-Dex1-Wholebody \
    --action_source dds_wholebody \
    --robot_type g129 \
    --enable_dex1_dds \
    --enable_wholebody_dds \
    --headless \
    --enable_cameras \
    "$@"
