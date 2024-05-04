{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import pandas as pd\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "def read_functional(paths):\n",
    "    df = []\n",
    "    for path in paths:\n",
    "        for i in range(path['conf_range'][0], path['conf_range'][1] + 1):\n",
    "            data_path = path['path'] + f'_{i:04}'\n",
    "            if(os.path.isfile(data_path)):\n",
    "                df1 = pd.read_csv(data_path)\n",
    "                if not df1.empty:\n",
    "                    df.append(df1)\n",
    "                    df[-1]['conf'] = i\n",
    "                    if 'copy' in df[-1]:\n",
    "                        if df[-1].loc[0, 'copy'] == 0:\n",
    "                            df[-1]['copy'] = df[-1]['copy'] + 1\n",
    "                    if 'parameters' in path:\n",
    "                            for key, val in path['parameters'].items():\n",
    "                                df[-1][key] = val\n",
    "    return pd.concat(df)\n",
    "\n",
    "def fill_funcational_max(df, groupby_keys):\n",
    "    df2 = []\n",
    "    copy_num = df.groupby(['copy']).ngroups\n",
    "    for copy_max in range(1, copy_num + 1):\n",
    "        df1 = df[df['copy'] <= copy_max]\n",
    "        df1 = df1.groupby(groupby_keys + ['conf'])['functional'].max().reset_index(level=groupby_keys + ['conf'])\n",
    "        df2.append(df1.groupby(groupby_keys)['functional']\\\n",
    "                   .agg([('functional', np.mean), ('std', lambda x: np.std(x, ddof=1)/math.sqrt(np.size(x)))]).reset_index(level=groupby_keys))\n",
    "        df2[-1]['copy'] = copy_max\n",
    "    return pd.concat(df2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
