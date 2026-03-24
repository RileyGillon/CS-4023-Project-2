How to Push Your Branch
git add .
git commit -m "Add your feature description"
git push origin feature/your-feature-name

---

## Connecting a TurtleBot 4 to the Ubuntu 24.04 Lab Computers

### Step 1 – Power on and position the robot

- Ensure the TurtleBot is powered on.
- Lift the robot **by its base** (not by its sensors or tower) and place it on the floor before operating it.
- Do not place the robot on elevated surfaces (desks, tables, etc.).

### Step 2 – Verify the robot's onboard computer is running

Both the desktop computer and the robot's Raspberry Pi connect to the Internet through the **WIFI@OU** wireless network. The robot's name is printed on the unit.

1. Open a terminal and SSH into the robot:
   ```bash
   ssh student@<nameFromRobot>.cs.nor.ou.edu
   ```
   The student account password is `student`.

2. Wait a few seconds, then list active ROS 2 topics:
   ```bash
   ros2 topic list
   ```
   Confirm that `/scan`, `/tf`, and `/odom` appear in the list.

3. If those topics are missing, restart the ROS 2 daemon:
   ```bash
   turtlebot4-daemon-restart
   ```
   Or manually:
   ```bash
   ros2 daemon stop
   ros2 daemon start
   ```
   Then run `ros2 topic list` again.

4. If topics are still missing after the robot finishes booting (startup chime), manually start the robot bringup:
   ```bash
   ros2 launch turtlebot4_bringup robot.launch.py
   ```

### Step 3 – Connect the desktop computer to the robot

Open a **new terminal on the desktop computer** (do not SSH into the robot) and run the setup script:

```bash
robot-setup.sh
```

When prompted, enter the TurtleBot's name. Then run the commands the script outputs, for example (for a robot named `matamata`):

```bash
unset ROS_LOCALHOST_ONLY
export ROS_DOMAIN_ID=8
export ROS_DISCOVERY_SERVER=";;;;;;;;10.194.16.61:11811;"
export ROS_SUPER_CLIENT=True
ros2 daemon stop
ros2 daemon start
```

Afterward, run `ros2 topic list` on the desktop terminal to confirm the robot's topics are visible.

> **Note:** These environment variables must be set in **every new desktop terminal** you open. Repeat this step for each additional desktop terminal you need.