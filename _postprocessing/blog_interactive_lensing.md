# Bending Starlight, by Hand

Gravitational lensing is one of those ideas that's easy to state and hard to *feel*: put enough mass between you and a distant galaxy, and spacetime curves so much that the galaxy's light gets smeared into arcs, rings, and multiple images. This little app lets you grab that galaxy and drag it around to watch it happen, live.

## What you're looking at

Two panels, side by side:

- **Left — the source plane.** Where the galaxy *actually* is. A plain glowing blob you can move with the mouse (or the sliders).
- **Right — the lensed sky.** What a telescope on Earth would *see* after a massive dark-matter halo (at redshift 1) bends the light on its way to us (from redshift 2).

Drag the source toward the center and watch single blobs split into arcs, snap into an **Einstein ring**, then break into multiple images. That's not a cartoon — every frame comes from photons traced through the curved spacetime of an NFW halo with EXCALIBUR, then mapped back to the sky.

## The knobs

Sliders reshape the background galaxy (position, size, ellipticity, Sérsic index, brightness). Checkboxes overlay the physics:

- **r_s** — the halo's scale radius,
- **θ_E** — the Einstein radius,
- **critical curve** — where magnification blows up (the bright arcs live here),
- **caustic** — its shadow in the source plane: cross it, and images appear or vanish.

## The fun part: five halos, same mass

A radio switch flips between five lens geometries — all the *same mass*, just shaped and oriented differently:

| Profile | What it is | The lesson |
|---|---|---|
| **Spherical** | a round halo | the textbook ring |
| **Elliptical** | a rugby ball lying sideways | ring → cross, caustic opens into a 4-cusp **astroid** |
| **Inclined** | the same ball, tilted 45° | the ellipse relaxes toward a circle |
| **Cigar ∥ LOS** | the ball pointed *at us* | looks round on the sky — but lenses **stronger** |
| **Triaxial** | three unequal axes, random tilt | the messy, realistic case: everything rotates |

The punchline is the **cigar**: end-on, it projects to a perfect circle, identical in symmetry to the sphere yet its Einstein ring is noticeably bigger. Same silhouette, more mass stacked along each sightline, stronger lensing. Lensing doesn't care about the 3D shape; it cares about what's piled up along your line of sight.

## Why bother

Real lenses are lumpy, triaxial, and randomly oriented. Playing with these idealized cases builds the intuition you need to read the real thing — and honestly, it's just fun to bend starlight with a mouse.

*Built with EXCALIBUR — Laurent Magri-Stella.*
