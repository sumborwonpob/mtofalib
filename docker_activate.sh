echo "For use inside docker. It will not run outside docker."

source /ros_entrypoint.sh
cd ../..
colcon build
source install/setup.bash