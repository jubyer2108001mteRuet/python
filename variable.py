{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "import random\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "%matplotlib inline\n",
    "%matplotlib widget\n",
    "from IPython.display import display\n",
    "from ipywidgets import interactive\n",
    "import ipywidgets as widgets\n",
    "\n",
    "# Training Data\n",
    "x_train = np.array([1.0, 2.0])\n",
    "y_train = np.array([300.0, 500.0])\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Data:**\n",
    "| Size (1000 sqft) | Price (1000s of dollars) |\n",
    "| -------------------| ------------------------ |\n",
    "| 1 | 300 |\n",
    "| 2 | 500 |\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Cost Function $ J(w, b) $:\n",
    "\n",
    "$$ J(w, b) = \\frac{1}{2m}\\sum_{i=0}^{i=m-1}(f_{w, b}(x^i) - y^i)^2 $$\n",
    "Here, $f_{w, b}(x^i) = wx^i+b$ \\\n",
    "$m$ is the number of training data points.\\\n",
    "If $f_{w, b}(x)$ is written as $y^{i_p}$ Then the equation becomes\n",
    "$$ J(w, b) = \\frac{1}{2m}\\sum_{i=0}^{i=m-1}(y^{i_p} - y^i)^2 $$\n",
    "Here, $0 \\to m-1$ is used because of the implementation in code. Actual mathematical representation is $1 \\to m$\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "def compute_cost(x, y, w, b):\n",
    "    \"\"\"\n",
    "    Computes the cost function of a linear regression.\n",
    "\n",
    "    Args:\n",
    "        x (ndarray (m,)): Data. Training inputs\n",
    "        y (ndarray (m,)): Data. Training outputs\n",
    "        w (scalar): Model weight\n",
    "        b (scalar): Model bias \n",
    "\n",
    "    Returns:\n",
    "        total_cost (float): The cost of using w, b as the parameters for linear regression to fit data points in x and y\n",
    "    \"\"\"\n",
    "    m = x.shape[0]  # number of training examples\n",
    "    cost_sum = 0\n",
    "    for i in range(m):\n",
    "        f_wbi = w * x[i] + b\n",
    "        cost = (f_wbi - y[i]) ** 2\n",
    "        cost_sum += cost\n",
    "    total_cost = (1/(2*m))*cost_sum\n",
    "\n",
    "    return total_cost\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b03cd61f8baf4f5b836228217fb8606e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "interactive(children=(IntSlider(value=200, description='w', layout=Layout(width='500px'), max=500, min=-500), …"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "min = 99999999\n",
    "def visualize_cost(x, y, w, b):\n",
    "    fig = plt.figure()\n",
    "    ax = plt.axes(projection ='3d')\n",
    "    global min\n",
    "    samples = 1000\n",
    "    w_list = np.linspace(-500,500, samples)\n",
    "    b_list = np.linspace(-500, 500, samples)\n",
    "    W_L, B_L = np.meshgrid(w_list, b_list)\n",
    "    C_L = np.zeros(shape=(samples, samples))\n",
    "    for i in range(samples):\n",
    "        for j in range(samples):\n",
    "            C_L[i][j] = compute_cost(x, y, W_L[i][j], B_L[i][j])\n",
    "    \n",
    "    ax.plot_surface(W_L,  B_L, C_L, cmap=\"plasma\")\n",
    "    cost = compute_cost(x, y, w, b)\n",
    "    ax.scatter(w, b, cost, color=\"black\")\n",
    "    # syntax for 3-D projection\n",
    "    \n",
    "    if(cost < min):\n",
    "        min = cost\n",
    "    # plt.scatter(w, cost, color=\"red\", marker=\"x\")\n",
    "\n",
    "    # ax.set_lab\n",
    "    plt.xlabel(\"Weights\")\n",
    "    plt.ylabel(f\"Biases\")\n",
    "    # ax.set_zlim([-2.0, 2.0])\n",
    "    # plt.zlabel(f\"Bias\")\n",
    "    plt.show()\n",
    "\n",
    "interactive_plot = interactive(visualize_cost, x=widgets.fixed(x_train), y=widgets.fixed(y_train), w=widgets.IntSlider(min=-500, max=500, value=200, layout=widgets.Layout(width='500px')), b=widgets.IntSlider(min=-500, max=500, value=200, layout=widgets.Layout(width='500px')))\n",
    "display(interactive_plot)"
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
   "display_name": "Python 3 (ipykernel)",
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
   "version": "3.11.1"
  },
  "vscode": {
   "interpreter": {
    "hash": "c6cd0b575fc0755e990455c55bafb95cd503212c1209f644472f7d8844f98f6e"
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}