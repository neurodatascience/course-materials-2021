# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from os.path import join
from bids import BIDSLayout
from bids.reports import BIDSReport
from bids.tests import get_test_data_path

# %%
layout = BIDSLayout(join(get_test_data_path(), 'synthetic'))
layout

# %%
layout.get_subjects()

# %%
layout.get_sessions()

# %%
layout.get_tasks()

# %%
f = layout.get(task='nback', run=1, extension='nii.gz')[0].filename
f

# %%
report = BIDSReport(layout)
report

# %%
counter = report.generate()
main_report = counter.most_common()[0][0]
print(main_report)

# %%
