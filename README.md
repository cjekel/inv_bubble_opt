# inv_bubble_opt

Tools for identifying material parameters from bulge inflation test data.

# Getting started

Clone this repo

```
git clone https://github.com/cjekel/inv_bubble_opt.git
```

and install the invbubble library

```
cd inv_bubble_opt
pip install invbubble_py_source/
```

# Experimental data

| Test Number | Material | File Name | Google Drive | Mega |
| ----------- | -------- | --------- |:------------:|:----:|
| 1 | VALMEX 7318 | blue00.npy | [link](https://drive.google.com/open?id=1gCdKFhzb8dr1UJBmg5Ywsjd6wAgc-nnn) | [link](https://mega.nz/#!NqoAVIxB!k4BnjtupDZwNhkmK9wiLcgEbJoYFfpZ4PT1ViSuH6WI) |
| 2 | VALMEX 7318 | blue01_rotated_90.npy | [link](https://drive.google.com/open?id=1ifOTVNmNcnaFtxnJc-HUFqWl6Fqh1P2E) | [link](https://mega.nz/#!I7gQkCbC!2Kys092LgEd553-yLRWreJGBARu92-8xnmhso0Sansw) |
| 3 | VALMEX 7318 | blue02_rotated_90.npy | [link](https://drive.google.com/open?id=1IBQVdEfEXL0e0pwrmGebZd7_TzPYW3hi) | [link](https://mega.nz/#!5nxiSAAA!OTfzmGRNgG3DuxxlMQz44FhjjWfGvHXlGini0P5beTU) |
| 4 | VALMEX 7318 | blue03.npy | [link](https://drive.google.com/open?id=1GFZQwc131NQS4CU0B5VEDIv9nc8QpMRC) | [link](https://mega.nz/#!lu5CwCbA!08Ubtocp95PvJrqozkyeCQiME2fJnQ9CedzbmGMoIDc) |
| 5 | CF0700T | black01.npy | [link](https://drive.google.com/open?id=1jtmWGAVcN4YFN42c2SUAOLZ17Q0FtQ6c) | [link](https://mega.nz/#!5jhQDCzA!Io7oGVVixFBv8IIe4o_NoOrScaoDi9IngE5NEi_15Mw) |
| 6 | CF0700T | black02.npy | [link](https://drive.google.com/open?id=11J-WHwiHXBXG-JWR1oke1aW_5kVYljIF) | [link](https://mega.nz/#!knw2zAJB!pv3Trcbd_7lGC9xgnXGCFLmfO7e-qFgVbC5Q5aYnpfU) |
| 7 | CF0700T | black03.npy | [link](https://drive.google.com/open?id=16zmo64WsyF5UTrcwz4tcZDzauCy1_7F2) | [link](https://mega.nz/#!hioUhIDJ!6PQGeX-MwP3Lb7rdB0i6pc_sGL0nAFvoR5BABe2jlJI) |
| 8 | CF0700T | black04.npy | [link](https://drive.google.com/open?id=180fhBiXFSOl6OpbJ9MHZJ0B24f9r5pH5) | [link](https://mega.nz/#!Uv4GzI4Y!PVCwLWaFM__ed9_CqXyRoZt7x4u4h-MeBeqatDaKpc4) |

The experimental data is stored as pickled binary numpy arrays. Note there is the potential for pickled binary numpy array to contain malware.

The follow code is an example on how to open and slice the data file.

```python
import numpy as np
blue00 = np.load('blue00.npy', allow_pickle=True)
i = 1
pressures = blue00[:, 1]  # inlation pressures in gigapascal (GPa)
x = blue00[i, 0][0]  # x locations at the ith pressure in mm
y = blue00[i, 0][1]  # y locations at the ith pressure in mm
z = blue00[i, 0][2]  # z locations at the ith pressure in mm
dx = blue00[i, 0][3]  # displace in x at the ith pressure in mm
dy = blue00[i, 0][4]  # displace in y at the ith pressure in mm
dz = blue00[i, 0][5]  # displace in z at the ith pressure in mm
```

# FE input decks

You must supply material values in place of the keyword in the *Material, name=Material-1 section in order for the input decks to run.

| Material | Parameters | Keywords | File location | link |
| -------- | ---------- | -------- | ------------- | ---- |
| isotropic | E | E1_iso | iso_one_param/model/model_template.inp | [link](https://github.com/cjekel/inv_bubble_opt/blob/master/iso_one_param/model/model_template.inp) |
| isotropic | E, nu | E1_iso, Nu12_iso | iso_two_param/model/model_template.inp | [link](https://github.com/cjekel/inv_bubble_opt/blob/master/iso_two_param/model/model_template.inp) |
| orthotropic | E1, E2, G12 | E1_orth, E2_orth, G12_orth, E23_orth | lin_ortho_known/model/model_template.inp | [link](https://github.com/cjekel/inv_bubble_opt/blob/master/lin_ortho_known/model/model_template.inp) |

