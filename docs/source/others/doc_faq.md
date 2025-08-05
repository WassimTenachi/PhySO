# FAQ

Frequently Asked Questions

---

### 1. I am having issues with LaTeX rendering

##### Running physo without LaTeX

If LaTeX is not detected on your system, `physo` should automatically disable LaTeX rendering (since it is an optional dependency).  
However, if LaTeX-related errors persist, you can manually disable LaTeX rendering by setting:

```python
FLAG_USE_LATEX_RENDERING = False
```
in the file:
`physo/physym/program.py`

##### Running physo with LaTeX
If you'd like to enable LaTeX rendering and are encountering issues, please ensure LaTeX is properly installed on your system.

- macOS
```bash
sudo apt install cm-super dvipng texlive-latex-extra texlive-latex-recommended
conda install latexcodec
```

- Linux
```bash
conda install -c conda-forge latexcodec
```

- Windows
```bash
conda install -c conda-forge miktex
```

Once installed, verify LaTeX availability with:
```python
import shutil
print(shutil.which('latex') is not None)
```
This should return True.

You can also test that LaTeX rendering in Matplotlib is working by running:
```python
import matplotlib as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
```

### 2. How can I support this project?

Consider citing `physo` in your work. You can find the citation information [here](https://physo.readthedocs.io/en/latest/r_reference.html).
If you are a researcher or a student, email me if you are interested in collaborating in the development of future features of `physo`.

### 3. I am encountering an issue that is not covered here

If you are facing an issue that is not addressed in this FAQ, update the package and its dependencies to their latest available versions.
You can then raise an issue on the [GitHub repository](https://github.com/WassimTenachi/PhySO/issues).
