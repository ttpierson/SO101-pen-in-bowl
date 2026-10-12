# Notes for driving the real SO-101

- This is a physical SO-101 arm on a table, not a simulation. Move deliberately: prefer several
  small moves with a look at the wrist camera in between over one large move.
- Action and `joint_pos` state order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll,
  gripper.
  - The five arm joints are in degrees, with 0 at the middle of each joint's calibrated range.
  - The gripper is 0..100: 0 is closed, 100 is fully open.
- The rest (folded) pose is about [0, -103, 103, 68, 81, 2]. In it, the arm is tucked down
  next to its base. Raising shoulder_lift toward 0 brings the upper arm up. Lowering elbow_flex
  toward 0 unfolds the forearm forward.
- shoulder_pan rotates the whole arm around the vertical axis.
  wrist_flex tilts the gripper up and down. wrist_roll rotates the gripper around its own axis.
- The only camera is on the wrist, looking out past the gripper jaws. Nothing gives you object
  positions; find the pen and the bowl in the wrist image. If you can't see them, raise
  the arm and pan slowly to look around.
- The pen is thin and lies flat on the table. Approach it from above with the jaws open,
  and turn wrist_roll so the jaws close across the pen, not along it. Close to grasp it.
  Lift, move over the bowl, lower a little, and open to drop it in.
- Joints are clamped to safe limits and every joint moves at most 3 degrees per control step.
  A large target therefore takes several steps to reach, so check the state before deciding the
  arm is stuck.
