#!/usr/bin/env python3
"""
Convert Niederreiter-NUT direction numbers to CSV format compatible with AQFED.
Format: d    s    a    m_i 

Writes niederreiter_a.csv and niederreiter_minit.csv to the current directory. The table below is only the start of
the data of this folder: do not run it from here, it would truncate the files read by ssoboldata.jl.
"""

import re
from pathlib import Path

# Parse the Niederreiter-NUT data
data_text = """d       s       a       m_i 
2 	1	0	1 
3 	2	1	1 3 
4 	3	1	1 1 5 
5 	3	2	1 3 7 
6	4	1	1 1 1 9 
7 	4	4	1 3 7 15 
8 	4	7	1 3 3 3 
9 	5	2	1 1 1 9 9 
10 	5	4	1 1 5 5 21 
11 	5	7	1 1 5 13 13 
12 	5	11	1 3 7 7 7 
13 	5	13	1 3 3 11 11 
14 	5	14	1 3 3 3 19 
15 	6	1	1 1 1 1 1 33 
16 	6	4	1 1 1 9 9 9 
17 	6	11	1 1 5 5 5 37 
18	6	13	1 1 5 13 29 61 
19 	6	16	1 3 7 15 31 63 
20 	6	19	1 3 7 15 15 15 
21	6	22	1 3 7 7 7 7 
22 	6	25	1 3 3 11 27 59 
23 	6	26	1 3 3 11 11 11 
24 	7	1	1 1 1 1 1 1 65 
25 	7	4	1 1 1 1 17 17 17 
26 	7	7	1 1 1 1 17 49 113 
27 	7	8	1 1 1 9 9 9 73 
28 	7	14	1 1 1 9 25 57 121 
29 	7	19	1 1 5 5 21 53 53 
30 	7	21	1 1 5 5 5 5 5 
31 	7	28	1 1 5 13 13 13 13 
32 	7	31	1 1 5 13 13 45 109 
33 	7	32	1 3 7 15 31 63 127 
34 	7	37	1 3 7 15 15 47 111 
35 	7	41	1 3 7 7 23 23 87 
36 	7	42	1 3 7 7 23 55 55 
37 	7	50	1 3 3 11 27 59 123 
38 	7	55	1 3 3 11 11 43 107 
39 	7	56	1 3 3 3 19 51 51 
40 	7	59	1 3 3 3 19 19 83 
41 	7	62	1 3 3 3 3 3 67 
42 	8	13	1 1 1 1 17 49 49 177 
43 	8	14	1 1 1 1 17 49 113 113 
44 	8	21	1 1 1 9 9 41 105 233 
45 	8	22	1 1 1 9 9 41 41 41 
46 	8	28	1 1 1 9 25 57 121 121 
47 	8	31	1 1 1 9 25 57 57 185 
48 	8	38	1 1 5 5 21 53 53 53 
49 	8	47	1 1 5 5 5 37 37 165 
50	8	49	1 1 5 13 29 29 29 29 
51 	8	50	1 1 5 13 29 29 93 221 
52 	8	52	1 1 5 13 29 61 61 189 
53 	8	56	1 1 5 13 13 13 13 141 
54 	8	59	1 1 5 13 13 13 77 77 
55	8	61	1 1 5 13 13 45 45 45 
56 	8	67	1 3 7 15 31 63 63 63 
57 	8	69	1 3 7 15 31 31 95 223 
58 	8	70	1 3 7 15 31 31 31 31 
59 	8	79	1 3 7 15 15 15 79 207 
60 	8	81	1 3 7 7 23 23 23 23 
61 	8	84	1 3 7 7 23 55 55 55 
62 	8	88	1 3 7 7 7 7 71 199 
63 	8	94	1 3 7 7 7 39 39 39 
64 	8	97	1 3 3 11 27 27 91 91 
65 	8	103	1 3 3 11 27 59 59 187 
66 	8	107	1 3 3 11 11 11 75 75 
67 	8	110	1 3 3 11 11 43 107 107 
68	8	115	1 3 3 3 19 51 115 243 
69 	8	121	1 3 3 3 3 35 99 227 
70 	8	122	1 3 3 3 3 35 35 35 
71 	8	124	1 3 3 3 3 3 67 195 
72 	9	1	1 1 1 1 1 1 1 1 257 
73 	9	8	1 1 1 1 1 33 33 33 33 
74 	9	11	1 1 1 1 1 33 33 161 417 
75 	9	13	1 1 1 1 1 33 97 97 353 
76 	9	16	1 1 1 1 17 17 17 17 273 
77 	9	22	1 1 1 1 17 17 81 209 465 
78	9	25	1 1 1 1 17 49 49 49 49 
79 	9	37	1 1 1 9 9 9 9 9 265 
80 	9	44	1 1 1 9 9 41 41 41 41 
81 	9	47	1 1 1 9 9 41 41 169 425 
82 	9	50	1 1 1 9 25 25 89 217 473 
83 	9	52	1 1 1 9 25 25 25 25 281 
84 	9	55	1 1 1 9 25 25 25 153 153 
85	9	59	1 1 1 9 25 57 121 249 249 
86 	9	62	1 1 1 9 25 57 57 185 441 
87 	9	67	1 1 5 5 21 21 85 213 213 
88 	9	74	1 1 5 5 21 53 117 245 501 
89 	9	76	1 1 5 5 21 53 53 53 309 
90 	9	81	1 1 5 5 5 5 69 69 69 
91 	9	82	1 1 5 5 5 5 69 197 453 
92	9	87	1 1 5 5 5 5 5 133 133 
93 	9	91	1 1 5 5 5 37 101 229 229 
94 	9	94	1 1 5 5 5 37 37 165 421 
95 	9	103	1 1 5 13 29 29 93 93 349 
96 	9	104	1 1 5 13 29 61 61 189 189 
97 	9	109	1 1 5 13 29 61 125 253 509 
98 	9	122	1 1 5 13 13 45 45 45 45 
99	9	124	1 1 5 13 13 45 109 237 237 
100 	9	128	1 3 7 15 31 63 127 255 511 
101	9	137	1 3 7 15 31 31 95 95 95 
102 	9	138	1 3 7 15 31 31 95 223 479 
103	9	143	1 3 7 15 31 31 31 159 415 
104 	9	145	1 3 7 15 15 47 47 175 175 
105 	9	152	1 3 7 15 15 15 15 15 271 
106 	9	157	1 3 7 15 15 15 79 79 335 
107 	9	164	1 3 7 7 23 23 87 215 215 
108 	9	167	1 3 7 7 23 23 87 87 343 
109 	9	173	1 3 7 7 23 55 119 119 375 
110 	9	176	1 3 7 7 7 7 71 199 455 
111 	9	181	1 3 7 7 7 7 7 135 391 
112 	9	182	1 3 7 7 7 7 7 7 7 
113 	9	185	1 3 7 7 7 39 103 103 103 
114 	9	191	1 3 7 7 7 39 39 167 423 
115 	9	194	1 3 3 11 27 27 91 91 91 
116 	9	199	1 3 3 11 27 27 27 27 27 
117 	9	208	1 3 3 11 11 11 11 139 395 
118 	9	218	1 3 3 11 11 43 43 171 427 
119 	9	220	1 3 3 11 11 43 107 107 107 
120 	9	227	1 3 3 3 19 51 51 179 179 
121 	9	229	1 3 3 3 19 51 115 115 371 
122 	9	230	1 3 3 3 19 51 115 243 243 
123 	9	234	1 3 3 3 19 19 19 19 275 
124 	9	236	1 3 3 3 19 19 83 211 211 
125	9	241	1 3 3 3 3 35 99 99 355 
126 	9	244	1 3 3 3 3 35 35 35 291 
127 	9	253	1 3 3 3 3 3 3 131 131 
"""

# Parse the data to extract 'a' coefficients and direction numbers
a_values = {}
minit_values = {}

lines = [line.strip() for line in data_text.strip().split('\n') if line.strip() and not line.startswith('d')]

for line in lines:
    parts = re.split(r'\s+', line)
    if len(parts) >= 4:
        try:
            d = int(parts[0])
            s = int(parts[1])
            a = int(parts[2])
            m_i = [int(x) for x in parts[3:]]
            
            a_values[d] = a
            minit_values[d] = m_i
        except ValueError:
            continue

# Sort by dimension
dimensions = sorted(a_values.keys())

# Write polynomial coefficients
a_csv_path = Path('niederreiter_a.csv')
with open(a_csv_path, 'w') as f:
    for d in dimensions:
        f.write(f"{a_values[d]}\n")

print(f"Created {a_csv_path} with {len(dimensions)} coefficients")

# Write direction numbers (similar to sobol_minit.csv format)
minit_csv_path = Path('niederreiter_minit.csv')
with open(minit_csv_path, 'w') as f:
    # Find max number of direction numbers needed
    max_m = max(len(v) for v in minit_values.values())
    
    # Write each row with all values
    for d in dimensions:
        m_vals = minit_values[d]
        # Pad with last value if needed to match sobol format
        while len(m_vals) < max_m:
            m_vals = m_vals + [m_vals[-1]]
        f.write(','.join(str(x) for x in m_vals) + '\n')

print(f"Created {minit_csv_path} with direction numbers ({len(dimensions)} dimensions, up to {max_m} values each)")
print(f"\nData parsed successfully:")
print(f"Dimensions: {min(dimensions)} to {max(dimensions)}")
print(f"Sample entries:")
for d in dimensions[:5]:
    print(f"  Dim {d}: a={a_values[d]}, m_i={minit_values[d]}")
