# AI-GP
Ganymede Systems

What to do right now

Go deep on theaigrandprix.com — technical specifications for the drones and the simulation 
platform will be shared at a later stage The AI Grand Prix, so check frequently for updates
and make sure your team is signed up for notifications.

Start building your autonomy stack — the three core areas to split up among your team are:

Computer vision — gate/obstacle detection from onboard camera feeds
Path planning — optimal trajectory through the course
Control/flight dynamics — translating planned paths into drone commands


Get comfortable with drone simulation — tools like Gazebo, AirSim, or even 
the PX4 SITL simulator are great to practice on before the official DCL platform
opens up. The closer to real drone physics, the better.

Study prior autonomous drone racing research  look into papers from the AlphaPilot 
challenge (Lockheed/DRL) and the work out of ETH Zürich on autonomous drone racing. 
Those teams solved very similar problems.

ssign roles now — with up to 8 members on a team, get clear ownership over 
perception, planning, and controls early so you're not duplicating work.


Week 1 2: Get your environment set up

Install Python (if not already), and get comfortable with NumPy, OpenCV, 
and a simulation environment. I'd recommend starting with AirSim or 
Gazebo with PX4 SITL  both simulate quadrotor physics well and 
let you test autonomy code without hardware. Pick a version control 
workflow (Git  GitHub) and get all teammates on it from day one.

Week 2 4: Build the three pillars in parallel
Divide your team across these three tracks:

Perception  detecting gates/obstacles from a camera feed. Start with classical 
OpenCV (color thresholding, contour detection) before jumping to neural nets. 
It's faster to prototype and you'll understand the failure modes better.
Path Planning  given known gate positions, compute an optimal trajectory. 
Your AE background in dynamics will directly apply here. Look into minimum snap 
trajectory generation  it's the standard in drone racing and is essentially just 
constrained polynomial optimization. Control  translating a desired trajectory 
into motor commands. A cascaded PID (position  velocity  attitude) is the standard 
starting point. Your controls coursework maps almost 1:1 here.

