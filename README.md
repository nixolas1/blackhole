# M87* particle simulation

An interactive WebGL 2 simulation of the glowing gas around M87*. For local use, clone the repository and open **index.html** in Chrome. [Launch the live simulator](https://blackhole-eta-two.vercel.app)

Play or pause with Space. Reset returns the particles and magnetic field to their initial state while preserving playback, speed and framing. Available speeds are 0.25×, 0.5×, 1×, 1.5× and 2×. **Particles** selects 125,000, 250,000 or 500,000 live parcels. The default is 125,000. Changing this setting splits parcels or samples mass-weighted representatives of the existing flow, preserving total mass and simulation time. The browser remembers the selected count when local storage is available. The simulation continues until paused or closed. The clock measures simulation seconds at the pace of the supplied clip.

[24-second preview at 2160 × 2160 and 60 fps](renders/black-hole-2160p60.mp4) · [Opening comparison](renders/comparison.png) · [Live simulator screenshot](renders/simulator.png)

## What the reference shows

The supplied clip opens with simulated emission around M87*. At approximately 5 seconds it begins morphing into the authors' GenDIReCT reconstruction from April 21, 2018. The authors used University of Illinois magnetically arrested disk simulations and relativistic radiative transfer at 230 GHz for the synthetic images. The illustrated 18° inclination and +0.94 spin describe this example. The study finds a ring diameter of 42.2 ± 1.5 μas and a preferred viewing inclination around 20°. Its imaging analysis leaves the spin magnitude unconstrained. [Nature paper and Methods](https://www.nature.com/articles/s41586-026-11241-9) · [Authors' article and animation](https://theconversation.com/is-this-iconic-black-hole-image-accurate-new-study-delivers-an-independent-result-296146)

## How it evolves

Every 1/60 simulation second, the GPU advances each gas parcel’s position, velocity and thermal state. A softened central potential produces differential rotation. Weak drag transports angular momentum, pressure gradients accelerate the gas, and collisional relaxation damps departures from circular flow. Fresh gas enters at the outer disk as parcels cross the capture boundary. The display projects the disk at 18° and uses the measured palette, brightness asymmetry and radial profile. A mild spatial sharpening filter defines the inner edge of the crescent.

The first 101 reference frames supply the opening emission and calibrate the moving gas. The fitter measures the azimuthal spectrum, how disturbances extend across neighboring radii, their lifetime in a rotating frame, and the distribution of emission at each point in the image. At five seconds the last measured disturbance continues rotating as fresh turbulent fluctuations develop with those statistics. Their phases and amplitudes are generated independently. The pattern drift varies by radius and follows the measured clockwise motion. Both intervals use the same particle heat response, palette and rendering kernel.

Each parcel deposits mass into a two-dimensional density field. Its change in density along the orbit changes its temperature through adiabatic compression or expansion with an effective adiabatic index of 1.5. This treats local disk thickness as fixed. Pressure responds to density gradients. Magnetic dissipation supplies additional heat and the gas cools toward its ambient temperature. Density and temperature both affect displayed emission.

Magnetic patches orbit at their local circular speed. Differential rotation winds their radial field into a toroidal component. Once the winding crosses a local instability threshold, the patch releases stored magnetic energy into a narrow sheet and its wider exhaust. Nearby gas heats, expands, shears into arcs and cools. The radiated excess travels with the gas. Reconnection is a plausible source of intermittent heating in magnetized accretion flows. Here it is a reduced model with calibrated visual timescales. Determining the cause of an individual bright patch would require the underlying fluid and radiation data. [Reconnection simulations](https://arxiv.org/abs/2109.15115) · [EKKO accretion shader](https://github.com/nixolas1/ekko/blob/51bd548151eff3440312bed83cd516523f977628/src/accretion.wgsl)

## Measurements and reproduction

[data/frames.csv](data/frames.csv) contains projected centroid, peak, color and motion measurements for all 161 frames of the supplied clip. [data/render-comparison.csv](data/render-comparison.csv) compares the 101 physical source frames with the corresponding instants in the 60 fps export. RGB errors are measured in source pixels whose brightest reference channel exceeds 24/255.

Python with NumPy, SciPy, opencv-python-headless, and Playwright, plus Chrome and FFmpeg, can regenerate the fit and preview on macOS.

```sh
python scripts/fit_reference.py --opening /path/to/black_hole.mp4
python scripts/fit_turbulence.py
python scripts/render.py --duration 24
python scripts/compare_render.py /path/to/black_hole.mp4
```
