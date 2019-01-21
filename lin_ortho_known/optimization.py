# MIT License

# Copyright (c) 2019 Charles Jekel

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


def write_material_model(x):
    with open('model/model_template.inp', 'r') as d:
        with open('model/model.inp', 'w') as f:
            data = d.read()
            data = data.replace('E1_orth', str(x[0]))
            data = data.replace('E2_orth', str(x[1]))
            data = data.replace('G12_orth', str(x[2]))
            data = data.replace('E23_orth', str(x[1]/2.48))
            f.write(data)


x = [800.0*1e-3, 150.0*1e-3, 25.0*1e-3]
write_material_model(x)
