> 来源：Christopher Bishop 的书《Pattern Recogniction and Machine Learning》第8章Graphical Models

## Chapter 8 Graphical Models

概率图模型（probabilistic graphical models）建立起了一套方法，不仅可以直观的表达随机变量之间关系，方便计算模型的概率分布，还能帮助设计新的模型。概率图模型类似于量子场论中反应粒子相互作用的费曼图（Feynman diagram），由费曼图可以计算粒子相互作用的散射截面。

概率图由结点（nodes）和边（edges）构成，每个结点表示一个随机变量（或者一组随机变量），每条边表示由其相连的随机变量之间的概率关系。由概率图，可以将图中所有随机变量的联合概率分布写成一些只依赖于部分随机变量的概率分布或条件概率分布的乘积，即概率图的因子化（factorization）。

概率图模型主要分为三种：

 - 有向图模型（directed graphical models），也称为贝叶斯网络（Bayesian networks），图中的边是有方向的箭头。有向图模型主要用于有因果关系的或时序模型。
- 无向图模型（undirected graphical models），也称为马尔科夫随机场（Markov random fields），图中的边没有方向。无向图模型更适合于表达有约束条件和相互作用的模型。
- 因子图模型（factor graph），将有向图和无向图结合起来的图模型。

### 8.1 贝叶斯网络（Bayesian Networks）

在有向图（即Bayesian network）中，结点表示随机变量，用箭头连接两个结点（nodes）。如果箭头从结点a指向结点b，则a是b的父结点，b是a的子结点。模型中事先确定的参数在有向图中用实心圆表示，观测量（observed variables）用阴影圆表示。不客观测量也称为潜在变量（latent variables）或隐变量（hidden variables）。

有向图有一个限制条件：从任意结点出发，沿着箭头的方向移动，不能返回到该结点。因此我们研究的有向图又称为非循环有向图（directed acyclic graphs, DAGs）。

以下图为例，随机变量a, b, c的联合概率为
$$
p(a, b, c) = p(c|a, b)p(b|a)p(a).
$$
<img src="figures/08-01-example-graph.png" style="zoom:50%" align=center />

有向图的联合概率的普遍的表达方式可以写成

$$
p(x_1, x_2, \cdots, x_K) = \prod_{i=1}^K p(x_i|\text{pa}_i) ,
$$

其中$\text{pa}_i$表示结点$x_i$的所有父结点。以看出有向图的联合概率分布是每个结点的条件概率的乘积，而每个结点的条件概率以其父结点为条件。这样表示的联合概率称为对应的有向图的因子化（factorization）。

> The joint distribution defined by a graph is given by the product, over all of the nodes of the graph, of a conditional distribution for each node conditioned on the variables corresponding to the parents of that node in the graph.

如果有向图中任意两个结点之间都有剪头连接，这种有向图称为全连通（fully connected）有向图。全连通有向图的联合概率可以写成下面的形式
$$
p(x_1, x_2, \cdots, x_K) = p(x_K|x_1, \cdots, x_{K-1}) \cdots p(x_2|x_1)p(x_1).
$$
全连通有向图对所有联合概率分布成立，如果在某个联合概率分布中相对全连通有向图的联合概率分布有一些项确实，反而会反映出这个联合概率分布中一些有意思的性质。

- **例：多项式回归**

在多项式回归中，观测数据集为$\mathbf{t}=(t_1, \cdots, t_N) ^\text{T}$，多项式的系数为$\mathbf{w}$，则对应的有向图可以表示成

<img src="figures/08-02-polynomial-regression.png" style="zoom:50%" align=center />

联合概率分布为 
$$
p(\mathbf{t}, \mathbf{w}|\mathbf{x}, \alpha, \sigma^2)=p(\mathbf{w}|\alpha)\prod_{n=1}^N p(t_n|\mathbf{w}, x_n, \sigma^2),
$$
上式中的$\mathbf{x}, \alpha, \sigma^2$是模型中事先确定的量，$\mathbf{t}$是观测量，$\mathbf{w}$是隐变量。

- **有向图用于生成模型（Generative models）**

<u>继承取样(ancestral sampling)</u>：在有向图中，将箭头的尾部对应的随机变量称为低层级结点（low-numbered node），箭头的头部对应的随机变量称为高层级结点（high-numbered node）。按照随机变量的层级从低到高取样，得到的样本称为继承取样；即从最低层级的结点出发，假设为$x_1$，按照其先验概率分布$p(x_1)$，取样结果为$\hat{x}_1$，然后对其高一层级的结点（假设为$x_2$）按照条件概率$p(x_2|\hat{x}_1)$取样得到$\hat{x}_2$，依次类推，对$x_n$按照条件概率$p(x_n|\hat{\text{pa}})$取样得到$\hat{x}_n$，最后得到所有随机变量按照有向图对应的联合概率分布的取样。

图模型可以按照上面的过程进行取样，而且样本中随机变量之间具有因果关系，因此可以称为生成模型（generative models）。

在概率模型中，一般高层级的结点对应的是观测量，低层级的结点对应的是隐变量，隐变量不一定需要有对应的物理意义。

- **有向图表示共轭概率分布（conjugate probability distribution）**

用有向图表示共轭概率分布时，父-子结点对对应的随机变量的分布相同。我们下面以离散分布（discrete probability distribution）和高斯分布为例，这两种分布都可以扩展到任意复杂的有向图。

1. **离散分布**

用1-of-K表示方法，离散分布的随机变量$\mathbf{x}$有$K$个状态，当$\mathbf{x}$处于状态k时，$x_k=1$，其他的$x_{i\neq k}=0$.

一个随机变量的离散分布为$p(\mathbf{x}|\mathbf{\mu})=\prod_{k=1}^{K}\mu_k^{x_k}$，其中参数$\mathbf{\mu}=(\mu_1, \cdots, \mu_K)^\text{T}$满足约束条件$\sum_{k=1}^K\mu_k=1$，所以共有$K-1$个参数。

两个随机变量的离散分布为$p(\mathbf{x}_1, \mathbf{x}_2|\mathbf{\mu})=\prod_{k=1}^K\prod_{l=1}^K\mu_{kl}^{x_{1k}x_{2l}}$，其中参数$\mathbf\mu$是2维张量，$\mu_{kl}^{x_{1k}x_{2l}}$表示$x_{1k}=1$且$x_{2l}=1$时$\mathbf\mu$的取值。$\mathbf\mu$满足约束条件$\sum_k\sum_l\mu_{kl}=1$，所以共有$K^2-1$个参数。

由此类推$M$个随机变量的离散分布共有$K^M-1$个参数，参数的数量随随机变量数量呈指数增长。实际上，这对应的是全连通的有向图。

1）下图中(a)表示两个随机变量离散分布的有向图。如果增加一个条件：两个随机变量相互独立，它们的离散分布为$p(\mathbf{x_1}, \mathbf{x_2}|\mathbf{\mu}, \mathbf{\nu})=p(\mathbf{x_1|\mathbf\mu})p(\mathbf{x_2}|\mathbf\nu)=\left(\prod_{k_1=1}^K\mu_{k_1}^{x_{k_1}}\right)\left(\prod_{k_2=1}^K\nu_{k_2}^{x_{k_2}}\right)$，其中$\mathbf\mu$和$\mathbf\nu$满足约束条件$\sum_{k=1}^K\mu_k=1$和$\sum_{k=1}^K\nu_k=1$，所以参数共有$2(K-1)$个。用图模型的语言说，就是通过去除掉下图(b)中两个结点之间的关系，减少了模型的参数个数，但是对模型的概率分布有了更多的约束。

<img src="figures/08-03-discrete-variables-1.png" style="zoom:50%" align=center />

2）对于$M$个随机变量的情况，还可以假设对应的有向图中相邻的随机变量才有关系，如下图

<img src="figures/08-04-discrete-variables-chain.png" style="zoom:50%" align=center />

对于第一个随机变量$\mathbf{x_1}$，先验概率$p(\mathbf{x_1})$有$K-1$个参数，之后的每个随机变量$\mathbf{x_i}$的条件概率$p(\mathbf{x_i}|\mathbf{x_{i-1}})$有$K(K-1)$个参数，即$K$个可能的$\mathbf{x_{i-1}}$，$\mathbf{x_i}$有$K-1$个参数。所以上图对应的模型中有$K-1+(M-1)K(K-1)$个参数，是$K$ 的二次函数，$M$的线性函数，降低了模型的复杂度。在有向图中显式的画出参数$\mathbf\mu$，见下图

<img src="figures/08-05-discrete-variables-chain-with-parameters.png" style="zoom:40%" align=center />

3）对于上图我们还可以进一步假设相邻的结点对应的条件概率$p(\mathbf{x_i}|\mathbf{x_{i-1}})$的参数相同，那么模型参数进一步减少成$K-1+K(K-1)=K^2-1$，对应的有向图为

<img src="figures/08-06-discrete-variables-chain-with-sharing-parameters.png" style="zoom:40%" align=center />

2. **线性高斯模型（linear-Gaussian models）**

考虑一个由D个随机变量构成的有向图，每个结点表示一个按高斯分布的随机变量，每个结点分布的均值是其父结点的线性组合，结点$i$的概率分布可以写成

$$
p(x_i|\text{pa}_i)=\mathcal{N}\left(x_i|\sum_{j\in\text{pa}_i}w_{ij}x_j+b_i, v_i\right),
$$
每个结点的概率分布乘起来得到整个模型的联合概率分布。对这个联合概率分布求对数，得到
$$
\ln p(\mathbf{x})=\sum_{i=1}^D\ln p(x_i|\text{pa}_i)
=-\sum_{i=1}^D\frac{1}{2v_i}\left(x_i-\sum_{j\in\text{pa}_i}w_{ij}x_j-b_i\right)^2+\text{const}.
$$
可以看出，这样得到的联合概率分布是一个多变量高斯分布。

下面用递归方法求联合概率分布的均值和方差。

每个节点$i$对应的随机变量按高斯分布，所以可以写成
$$
x_i = \sum_{j\in\text{pa}_i}w_{ij}x_j+b_i+\sqrt{v_i}\epsilon_i,
$$
其中$\epsilon_i$是均值为0，方差为1的高斯分布的随机变量，即满足条件 $\mathbb{E}[\epsilon_i]=0$ 和 $\mathbb{E}[\epsilon_i\epsilon_j]=I_{ij}$，$I_{ij}$是单位矩阵。从层级低的结点依次计算均值，可以得到所有随机变量的均值，
$$
\mathbb{E}[x_i] = \sum_{j\in\text{pa}_i}w_{ij}\mathbb{E}[x_j]+b_i.
$$
同样的，可以从层级低的结点开始，依次计算随机变量$x_i$和$x_j$的协方差，
$$
\begin{align}
\text{cov}[x_i,x_j] &= \mathbb{E}\left[(x_i-\mathbb{E}[x_i])(x_j-\mathbb{E}[x_j])\right]\\
&=\mathbb{E}\left[(x_i-\mathbb{E}[x_i])\left\{\sum_{k\in\text{pa}_j}w_{jk}(x_k-\mathbb{E}[x_k])+\sqrt{v_j}\epsilon_j\right\}\right]\\
&=\left(\sum_{k\in\text{pa}_j}w_{jk}\mathbb{E}[(x_i-\mathbb{E}[x_i])(x_k-\mathbb{E}[x_k])]\right)+\sqrt{v_j}\mathbb{E}[\epsilon_j(x_i-\mathbb{E}[x_i])]\\
&=\left(\sum_{k\in\text{pa}_j}w_{jk}\text{cov}[x_i, x_k]\right)+\sqrt{v_j}\mathbb{E}[\epsilon_jx_i]\\
&=\left(\sum_{k\in\text{pa}_j}w_{jk}\text{cov}[x_i, x_k]\right)+\sqrt{v_j}\mathbb{E}[\epsilon_j\epsilon_i]\\
&=\left(\sum_{k\in\text{pa}_j}w_{jk}\text{cov}[x_i, x_k]\right)+v_jI_{ij}.
\end{align}
$$

上面递归地求高斯分布随机变量的均值和方差，下面讨论两个简化的特例。

1）有向图中所有结点之间的连接都去除，即各高斯分布的随机变量之间相互独立。这种情况下，$w_{ij}=0$，模型中的参数只剩下$D$个$b_i$和$D$个$v_i$，总参数个数为$2D$。

2）全连通有向图中，某个结点的父结点是所有比它层级低的结点。这种情况下，$w_{ij}$的第$i$行有$i-1$个非零元素，这样构成了一个下三角矩阵，因此$w_{ij}$有$(D^2-D)/2=D(D-1)/2$个参数，再加上$D$个$v_i$，总共有$D(D+1)/2$个参数。 

### 8.2 条件独立性（Conditional independence）

- **条件独立性的定义**

可以有两种定义方式：

1）假设有三个随机变量$a$，$b$和$c$，如果满足$p(a|b,c)=p(a|c)$，则称在给定随机变量$c$的条件下，$a$和$b$是独立的。

2）假设有三个随机变量$a$，$b$和$c$，如果满足$p(a,b|c)=p(a|b,c)p(b|c)=p(a|c)p(b|c)$，则称在给定随机变量$c$的条件下，$a$和$b$是独立的。

如果给定随机变量$c$的条件下，随机变量$a$和$b$是独立的，则记为$a\!\perp\!\!\!\perp b\mid c$.

如果给定随机变量$c$的条件下，随机变量$a$和$b$是独立的，则记为$a \not\!\perp\!\!\!\perp b \mid c$

- **三个典型的条件独立性例子**

以下讨论三个随机变量的几种条件独立性。

1）尾尾连接（tail-to-tail）：有向图中，随机变量$c$与$a$和$b$是用箭头的尾部连接的。联合概率分布为
$$
p(a,b,c)=p(a|c)p(b|c)p(c).
$$
首先讨论非条件概率的情况，
$$
p(a,b)=\sum_c p(a,b,c)=\sum_c p(a|c)p(b|c)p(c)\neq p(a)p(b),
$$
所以$a$和$b$不是相互独立的随机变量，即$a\not\!\perp\!\!\!\perp b|\varnothing$.

<img src="figures/08-07-tail-to-tail.png" style="zoom:40%" align=center />

然后讨论相对$c$的条件概率，即$c$为观测量，在已经观测到$c$的情况下，考虑$a$和$b$是否相互独立，
$$
p(a,b|c)=\frac{p(a,b,c)}{p(c)}=\frac{p(a|c)p(b|c)p(c)}{p(c)}=p(a|c)p(b|c),
$$
所以在给定$c$的条件下，$a$和$b$是相互独立的，即$a\!\perp\!\!\!\perp b |c$.

2）头尾连接（head-to-tail）：有向图中，随机变量$c$与$a$和$b$之间，一个以箭头的头部连接，另一个以箭头的尾部连接。联合概率分布为
$$
p(a,b,c)=p(a)p(c|a)p(b|c).
$$
首先讨论非条件概率的情况，
$$
p(a,b)=\sum_c p(a,b,c)=p(a)\sum_c p(b|c)p(c|a)=p(a)p(b|a)\neq p(a)p(b|a),
$$
所以$a$和$b$不是相互独立的随机变量，即$a\not\!\perp\!\!\!\perp b|\varnothing$.

<img src="figures/08-08-head-to-tail.png" style="zoom:40%" align=center />

然后讨论相对$c$的条件概率，即$c$为观测量，在已经观测到$c$的情况下，考虑$a$和$b$是否相互独立，
$$
p(a,b|c)=\frac{p(a,b,c)}{p(c)}=\frac{p(a)p(c|a)p(b|c)}{p(c)}=\frac{p(a,c)p(b|c)}{p(c)}=p(a|c)p(b|c),
$$
所以在给定$c$的条件下，$a$和$b$是相互独立的，即$a\!\perp\!\!\!\perp b |c$.

3）头头连接（head-to-head）：有向图中，随机变量$c$与$a$和$b$是用箭头的尾部连接的。联合概率分布为
$$
p(a,b,c)=p(a)p(b)p(c|a,b).
$$
首先讨论非条件概率的情况，
$$
p(a,b)=\sum_c p(a,b,c) = p(a)p(b)\sum_c p(c|a,b)=p(a)p(b),
$$
所以$a$和$b$是相互独立的，即$a\!\perp\!\!\!\perp b |\varnothing$.

<img src="figures/08-09-head-to-head.png" style="zoom:40%" align=center />

然后讨论相对$c$的条件概率，即$c$为观测量，在已经观测到$c$的情况下，考虑$a$和$b$是否相互独立，
$$
p(a,b|c)=\frac{p(a,b,c)}{p(c)}=\frac{p(a)p(b)p(c|a,b)}{p(c)}\neq p(a)p(b),
$$
所以在给定$c$的条件下，$a$和$b$不是相互独立的，即$a\not\!\perp\!\!\!\perp b |c$.

头头连接的情况还可以推广，定义如果结点$y$可以从结点$x$出发，沿着箭头的方向到达，则称结点$y$是结点$x$的后继（descendant）。头头连接的情况推广成：如果头头连接的结点和它的任意一个后继结点是观测量（即作为条件），那么起初的结点条件不独立。

在图模型中，条件独立结点也称为被阻断（blocked），不独立的结点称为未阻断（unblocked）。

**小结**：

有向图中，尾尾连接和头尾连接的结点本身不独立，以所连接结点为观测量的情况下条件独立。头头连接的情况与前两者相反。

- **D-阻隔（D-seperation）**

D-阻隔中的D是directed的缩写，表示是有向图中的条件独立性（被阻隔）。

假设一个有向图的结点可以分成互不相交的三个集合$A, B, C$，它们的交集可能比整个有向图要小。考虑所有从$A$中任意结点到$B$中任意结点的路径，如果所有这样的路径中的结点满足下面的两个条件之一，

(a) 路径中在这个结点是头尾连接或者尾尾连接，并且这个结点在$C$中，或者

(b) 路径中在这个结点是头头连接，并且这个结点和它的后继结点都不在$C$中，

那么称为这条路径被阻断了(blocked)。

如果所有路径都被阻断，则称$A$被$C$从$B$中d-阻隔（d-separated），并且这个有向图中所有随机变量的联合概率满足$A\!\perp\!\!\!\perp B|C$.

例1：下图中从结点 $a$ 到 $b$的路径中，$f$是尾尾连接，而且不是观测量，所以$f$没有阻断。$e$是头头连接，$e$本身不是观测量，但是它的后继$c$是观测量，所以$e$也没有阻断。因此$a\not\!\perp\!\!\!\perp b|c$.

<img src="figures/08-10-d-separation-1.png" style="zoom:40%" align=center /> 

例2:下图中从结点$a$到$b$的路径中，$f$是尾尾连接，而且它是观测量，所以$f$阻断了。$e$是头头连接，而且$e$和它的后继$c$都不是观测量，所以$e$也阻断了。因此$a\!\perp\!\!\!\perp b|f$.

<img src="figures/08-11-d-separation-2.png" style="zoom:40%" align=center />

例3：独立同分布（i.i.d.）数据取样

独立同分布数据集$\mathcal{D}=\{x_1,\cdots,x_N\}$可以用下面的有向图表示，用d-separation的观点看，从任意一个样本$x_i$到$x_j$，$i, j\in[1, N]$，的路径中，$\mu$是尾尾连接，而且$\mu$不是观测量，所以$\mu$阻断了连接，即$x_i\!\perp\!\!\!\perp x_j |\mu, i\neq j$. 

给定$\mu$的条件下，所有样本数据对的条件概率分布为$p(\mathcal{D}|\mu)=\prod_{n=1}^Np(x_n|\mu)$. 但是如果不给定$\mu$，将条件概率与$\mu$的先验分布相乘，得到样本数据与$\mu$的联合概率分布，然后将其中的$\mu$ marginalize，得到样本数据的概率分布为$p(\mathcal{D})=\int_0^\infty p(\mathcal{D} |\mu)p(\mu)d\mu \neq\prod_{n=1}^Np(x_n)$，所以样本数据之间不是相互独立的。

<img src="figures/08-12-d-separation-idd.png" style="zoom:40%" align=center />

例4：朴素贝叶斯模型(naive Bayes model)

朴素贝叶斯模型可以用于分类。假设观测量用一个$D$维随机变量表示，$\mathbf{x}=(x_1, \cdots, x_D)^\text{T}$，分类的类别用1-of-k表示方法表示成一个$K$维二元矢量$\mathbf{z}$。

朴素贝叶斯模型的主要假设是给定分类$\mathbf{z}$，输入变量$x_1, \cdots, x_D$是条件独立的。

给定$\mathbf{z}$后，因为$x_i$ 和$x_j$ ($i\neq j$)之间是尾尾连接的，所以$\mathbf{z}$ 阻断了连接，所以$x_i\!\perp\!\!\!\perp x_j |\mathbf{z}, i\neq j$.

<img src="figures/08-13-naive-bayes.png" style="zoom:40%" align=center />

所有观测量的条件概率分布为$p(\mathcal{D}|\mathbf{z})=\prod_{n=1}^Np(x_n|\mathbf{z})$，计算观测量的概率分布为$p(\mathcal{D})=\int_0^\infty p(\mathcal{D} |\mathbf{z})p(\mathbf{z})d\mathbf{z} \neq\prod_{n=1}^Np(x_n)$，所以观测量之间不是相互独立的。

因此，在每个给定的分类$\mathbf{z}$中，该分类下的观测量的协方差矩阵是对角阵，但是所有观测量的协方差矩阵不是对角阵，而是由各个分类的对角协方差矩阵的叠加。

朴素贝叶斯模型虽然对观测量之间做了条件独立性的简化，但是在1）输入空间维度$D$很高的情况，2）输入数据包含有离散分布的数据和连续分布的数据时，能够给出较好的近似和处理。

- 从另外一个角度看，有向图可以看成是概率分布的一个过滤器（filter），有两种理解方式：

1) 对于一个联合概率分布$p(\mathbf{x})$，如果它满足一个有向图的因子化（factorization）公式，那么它就通过了这个有向图的过滤。所有满足一个有向图过滤条件（因子化公式）的概率分布组成了一个集合，称为有向因子化集合（directed factorization，记作$\mathcal{DF}$）。

2) 另外一种理解方式是首先将所有满足一个有向图要求的D-阻隔的条件独立性列出来，然后保留满足这些条件独立性的所有概率分布，这样组成的概率分布的集合也是有向因子化集合（$\mathcal{DF}$）。

举两个极端情况的例子，1）全连通有向图中没有条件独立性，所以对应的$\mathcal{DF}$是有向图的结点对应随机变量的任意联合概率分布的集合；2）没有边的有向图中所有随机变量都是独立的，所有对应的$\mathcal{DF}$是有向图的结点对应随机变量各自概率分布的乘积。

注意满足一个有向图的过滤条件的$\mathcal{DF}$包含了那些满足因子化条件，而且额外满足其他条件独立性的概率分布。

- **Markov blanket 或 Markov boundary**

假设一个由$D$个结点组成的有向图的联合概率分布$p(\mathbf{x}_1, \cdots, \mathbf{x}_D)$，考虑其中一个结点对应的随机变量$\mathbf{x}_i$，在给定其他所有结点对应的随机变量$\mathbf{x}_{j\neq i}$的条件下的条件概率分布。利用乘法规则和有向图的因子化公式，得到
$$
\begin{align}
p(\mathbf{x}_i|\mathbf{x}_{\{j\neq i\}}) &= \frac{p(\mathbf{x}_1, \cdots, \mathbf{x}_D)}{\int p(\mathbf{x}_1, \cdots, \mathbf{x}_D)d\mathbf{x}_i}\\
&= \frac{\prod_k p(\mathbf{x}_k|\text{pa}_k)}{\int \prod_k p(\mathbf{x}_k|\text{pa}_k)d\mathbf{x}_i},
\end{align}
$$
在上式分母的积分项中，与$\mathbf{x}_i$无关的条件概率项可以提到积分之外，然后分子和分母中相抵消剩余的项中都是与$\mathbf{x}_i$有关的条件概率项，包括$p(\mathbf{x}_i|\text{pa}_i)$本身和以$\mathbf{x}_i$为父结点的项。以$\mathbf{x}_i$为父结点的项包括$\mathbf{x}_i$的子结点和$\mathbf{x}_i$的子结点的其他父结点（co-parents）。

因此，==有向图中某个结点的Markov  blanket包括它的父结点、子结点和子结点的其他父结点（co-parents）==。这里需要包含子结点的其他父结点（co-parents）是因为explaining away现象，也就是说在多因一果的事件中，子结点没有阻隔co-parents，考虑的结点与子结点的其他父结点（co-parents）并不是条件独立的。Markov blanket是有向图中与某个结点相关的结点的集合，其他结点与这个结点隔绝开来。下图表示结点$\mathbf{x}_i$的Markov blanket。

<img src="figures/08-14-markov-blanket-directed-graph.png" style="zoom:40%" align=center />



### 8.3 马尔科夫随机场（Markov Random Fields）

无向图（undirected graph）是由表示随机变量或一组随机变量的结点和连接结点的边组成，这些边是没有方向性的。无向图也称为马尔科夫随机场（Markov random field）或马尔科夫网络（Markov network）。

- **无向图的条件独立性**

无向图的条件独立性比有向图的形式更简单。由于连接结点的边没有方向性，所以也就没有子结点和父结点这样的不对称性。

假设无向图中可以归纳出三组结点$A$、$B$和$C$，如果从$A$中任意结点到$C$中任意结点的路径必须经过一个或多个$B$中的结点，则称给定$B$的情况下，$A$和$C$是条件独立的，即$B$将$A$和$C$阻隔开了。如下图所示。

<img src="figures/08-15-undirected-graph-conditional-independence.png" style="zoom:40%" align=center />

无向图中某个结点的Markov blanket是所有与之相连的结点，比有向图中某个结点的Markov blanket要简单，如下图所示。

<img src="figures/08-16-markov-blanket-undirected-graph.png" style="zoom:40%" align=center />

- **无向图的因子化（factorization）性质**


无向图的因子化也是将无向图上随机变量的联合分布表示成一组反映图中表示局域性质的函数的乘积。

无向图可以看成是由一些局域的==块（clique）==组成的，每个clique是由一些全连接的结点组成的子图。一个结点也可以成为一个clique。==最大块（maximal clique）==是能够构成clique的最大的clique，如果在其中添加任何其他的结点，就不再构成clique。下图中含有两个结点的clique有$\{x_1, x_2\}$，$\{x_1, x_3\}$，$\{x_2, x_3\}$，$\{x_2, x_4\}$和$\{x_3, x_4\}$。$\{x_1, x_4\}$之间没有连接，不是clique。含有三个结点的clique有，$\{x_1, x_2, x_3\}$和$\{x_2, x_3, x_4\}$。含有两个结点的clique都不是maximal clique，含有三个结点的clique都是maximal clique。

<img src="figures/08-17-clique.png" style="zoom:40%" align=center />

<u>无向图的因子化</u>：联合概率分布可以分解成无向图中的maximal clique对应的函数的乘积。
$$
p(\mathbf{x}) = \frac{1}{Z}\prod_C \psi_C (\mathbf{x}_C),
$$
其中$\mathbf{x}$表示无向图中结点对应的随机变量；$C$表示maximal clique；$\mathbf{x}_C$表示maximal clique中结点对应的随机变量；$\psi_C$称为势函数（potential function）；$Z$称为配分函数（partition function），是概率的归一化充数，
$$
Z=\sum_\mathbf{x}\prod_C\psi_C(\mathbf{x}_C).
$$
注意：

1) 因为非maximal clique是maximal clique的子集，所以maximal clique的势函数包含了非maximal clique的势函数，所以在无向图的因子化公式中没有必要重复的列出非maximal clique的势函数，而只需要包含所有maximal clique的势函数的乘积。

2) 由于势函数的乘积要表示一个概率，所以要求势函数$\psi_C(\mathbf{x}_C)\geqslant 0$，除此之外，对势函数的形式没有其他过多要求；而在有向图因子化中，要求每个乘积都是一个条件概率或marginal概率。

3) 配分函数是无向图因子化中的主要困难。用N个互相不连通结点构成的无向图为例，假设每个结点是有K个状态的离散随机变量，那么求配分函数需要计算$K^N$项的和，计算量与结点数呈指数关系。对于要计算无向图模型的整体联合概率的问题，不可避免的要计算配分函数，计算量会很大；但是对于研究无向图模型中局域的条件概率问题，由于条件概率是两个marginal概率相除，配分函数在分子和分母中会抵消，所以这种问题中不用计算配分函数，计算量会小很多。

<u>无向图的条件独立性和因子化之间的关系</u>：

在势函数严格大于零（$\psi_C>0$）的条件下，定义无向独立性集合$\mathcal{UI}$（undirected independence）是所有满足无向图的条件独立性的概率分布的集合，另外定义无向因子化集合$\mathcal{UF}$（undirected factorization）是满足无向图中maximal clique对应的因子化公式的所有概率分布。

根据**Hammersley-Clifford theorem**，$\mathcal{UI}$ 和 $\mathcal{UF}$是等价的。

类比统计热力学中描述热力学系统中不同能量上状态数的Boltzmann分布，我们可以把势函数写成Boltzmann分布的形式，
$$
\psi_C(\mathbf{x}_C)=e^{-E(\mathbf{x_C})},
$$
其中$E(\mathbf{x}_C)$称为能量函数（energy function）。借用统计热力学中的说法，无向图的总能量是所有maximal cliques的能量之和。

无向图因子化过程中的势函数或能量函数的选取比较麻烦，需要整体考虑各个maximal clique之间的影响。

- **例：图像降噪（de-noising）**

问题：有一幅二元像素的图像，每个像素$x_i\in \{-1, +1\}$，共有$D$个像素，这是原始图像。另外随机地把原始图像中的10%像素的状态翻转，引入噪声，这是噪声图像，其中每个像素$y_i\in \{-1, +1\}$。我们的目标是对噪声图像进行降噪，使之尽可能恢复到原始图像接近的状态。

根据问题本身的信息，由于噪声图像中只有10%的像素被翻转，所以可以知道原始图像的像素$x_i$和噪声图像的像素$y_i$之间有很强的关联性。另外噪声图像中的相邻像素$x_i$和$x_j$之间的相关性也很强。我们可以用无向图来描述这个问题，如下图。下图中$y_i$是噪声图像的像素，为观测量。

<img src="figures/08-18-image-de-noising-undirected-graph.png" style="zoom:40%" align=center />

上图中的maximal cliques是每个$\{x_i, y_i\}$对和相邻的$\{x_i, x_j\}$对。对于$\{x_i, y_i\}$对，选取能量函数为$-\eta x_i y_i$，其中$\eta$为正常数。当$x_i$和$y_i$符号相同时，对应的能量最低，反之则能量最高。对于相邻的$\{x_i, x_j\}$对，选取能量函数为$-\beta x_ix_j$，其中$\beta$为正常数，同样的，相邻的$x_i$和$x_j$符号相同时，对应能量最低，反之能量最高。整体来看，系统趋于能量最低的状态。另外，在整体能量函数中加上$hx_i$项，表示整个图像更偏向于某一个状态。因此整体能量函数为
$$
E(\mathbf{x}, \mathbf{y}) = h\sum_i x_i -\beta\sum_{\{i,j\}}x_i x_j - \eta\sum_i x_i y_i,
$$
而因子化的联合概率分布为
$$
p(\mathbf{x}, \mathbf{y}) = \frac{1}{Z}\exp\{-E(\mathbf{x}, \mathbf{y}\}.
$$
在已知噪声图像$\mathbf{y}$的情况下，实际上计算的是条件概率$p(\mathbf{x}|\mathbf{y})$，即已知噪声图像还原原始图像的概率。在统计物理中，这个模型是Ising model。

求解$p(\mathbf{x}|\mathbf{y})$极大值也就是求解能量的极小值。可以采用iterated conditional modes （ICM）技术求解局域极小值。首先将原始图像$\{x_i\}$初始化，例如初始化为与噪声图像相同的值，$x_i = y_i，\forall  i$。然后选取一个像素$x_i$，固定其他像素，比较$x_i$分别取$+1$ 和 $-1$ 时，系统整体能量孰低，选取其中较低赋值给$x_i$，依次方式每次选取一个像素求局域能量最低，这样反复的进行迭代，最终会收敛到整体能量的局域极小值。每次选取一个像素的方法可以是按照规律依次选取，也可以随机选取。

求解$p(\mathbf{x}|\mathbf{y})$的极大值还可以用其他方法，例如max-product algorithm一般能得到更好的局域极小值，另外对于某些特殊情况，graph cuts方法可以求解全局极小值。

- **无向图与有向图的关系**

讨论有向图和无向图之间的互相转换。

<u>有向图转换成无向图</u>：

1. 有向图中结点只有一个父结点的情况：

<img src="figures/08-19-directed-to-undirected-graph-1-parent-node.png" style="zoom:40%" align=center />

上图（a）有向图中除第一个结点外，其他结点都只有一个父结点，按照有向图因子化公式，联合概率分布为
$$
p(\mathbf{x})=p(x_1)p(x_2|x_1)p(x_3|x_2)\cdots p(x_N|x_{N-1}),
$$
把箭头的方向去掉，就变成了无向图（b），按照无向图因子化公式，联合概率分布为
$$
p(\mathbf{x}) = \frac{1}{Z}\psi_{1,2}(x_1, x_2)\psi_{2,3}(x_2,x_3)\cdots\psi_{N, N-1}(x_N, x_{N-1}).
$$
很容易看出，两种联合概率分布可以直接对应起来
$$
\begin{align}
\psi_{1,2}(x_1,x_2) &= p(x_1)p(x_2|x_1),\\
\psi_{2,3}(x_2,x_3) &= p(x_3|x_2),\\
\vdots\\
\psi_{N, N-1}(x_N|x_{N-1}) &= p(x_N|x_{N-1}).
\end{align}
$$
由于无向图因子化的势函数直接对应有向图因子化中的条件概率，所以这样选取的势函数的配分函数$Z=1$.

2. 有向图中结点有多个父结点的情况：

<img src="figures/08-20-directed-to-undirected-graph-3-parent-nodes.png" style="zoom:40%" align=center />

上图（a）中的结点$x_4$有3个父结点$x_1, x_2,x_3$，按照有向图因子化公式，联合概率分布为
$$
p(\mathbf{x}) = p(x_1)p(x_2)p(x_3)p(x_4|x_1, x_2, x_3).
$$
其中含有$p(x_4|x_1,x_2,x_3)$，它是4个结点对应随机变量的函数。对应到无向图中，要求$x_1,x_2,x_3,x_4$这4个结点在一个clique中，所以将（a）中的有向图对应成无向图时，需要把箭头的方向去掉之外，还要把3个父结点连接起来，构成一个4个结点的全连接clique，如图（b）。这种操作相当于把父结点互相“结婚”，称为moralization，对应的无向图称为moral graph. 

利用moral graph将有向图转换成无向图的过程中丢失了一些条件独立性，例如上面一个结点有三个结点的例子中，$x_1,x_2,x_3$之间时头头连接，所以$x_1\perp\!\!\!\perp x_2 | x_4, x_1\perp\!\!\!\perp x_3 | x_4, x_2\perp\!\!\!\perp x_3 | x_4$，但是这些条件独立性在moral graph中不成立。

有向图转换成无向图的过程中，如果不考虑条件独立性，可以将有向图中的结点进行全连通，这样得到的无向图丢失了有向图中所有的条件独立性。使用moral graph方法将有向图转换成无向图，得到的无向图尽最大可能保留了有向图中的条件独立性，但还是丢失了一些条件独立性。

综上，<u>将有向图转换成无向图的方法</u>：

1）首先将有向图中每个结点的父结点之间连接起来，并且将有向图的箭头方向去掉，这样就得到了moral graph；

2）然后将moral graph中的clique的势函数初始化为1，再将有向图中的条件概率乘到对应的clique的势函数上。一般总能找到maximal clique尽可能用较少的maximal clique来将无向图因子化。

3）因为无向图因子化的势函数都是用有向图中的条件概率或marginal概率对应过来，所以配分函数Z总是等一1.

<u>无向图转换成有向图</u>：

无向图转换成有向图一般不太常用，其中归一化的约束是比较难解决的问题。但是在精确推断技术（exact inference techniques）中无向图转换成有向图起到很重要的作用，例如结合树算法（junction tree algorithm）。

<u>图（有向图或无向图）与分布的关系</u>：

1）D map（dependency map）：如果一个概率分布中含有的所有条件独立性都反应在一个图中，则这个图称为这个概率分布的D map。完全不连通图是任意概率分布的trivial D map。

2）I map（independence map）：如果一个图中表示的所有条件独立性都体现在一个概率分布中，则这个图称为这个概率分布的I map。全连通图是任意概率分布的trial I map。

3）perfect map：如果一个概率分布中含有的所有条件独立性都能在一个图中反应，而且该图中的所有条件独立性也体现在该概率分布中，则这个图称为这个概率分布的perfect map。perfect map既是D map，也是I map。

假设能够对应于一个perfect map的所有概率分布的集合是$P$，见下图。perfect map对应于一个有向图的所有概率分布的集合是$D$，perfect map对应于一个无向图的所有概率分布的集合是$U$。那么$D$和$U$都是$P$的真子集。也就是说，在所有能找到对应的perfect map的概率分布中，有些只能对应于有向图，有些只能对应于无向图，还有一些概率分布没有对应的有向图或无向图作为perfect map。

<img src="figures/08-21-directed-undirected-graphs-and-perfect-map.png" style="zoom:40%" align=center />

在概率图模型框架中，还可以推广到在一个图中同时拥有有向图部分和无向图部分，这样的图称为链图（chain graphs）。但是还是有一些概率分布的perfect map无法用链图表示。本书不对链图做进一步分析。




### 8.4 图模型中的推断（Inference in Graphical Models）

图模型可以用直观的形式帮助我们进行统计推断。图模型上的很多算法可以表示城市图中一些局域的信息（messages）在图中的传播。

本节讲解exact inference techniques，第10章讲解approximate inference algorithms。

- **链图上的推断（Inference on a chain）**

​    考虑下面无向链图上的推断问题，即求解链图上某个或某些结点的marginal概率。值得注意的是，因为有向链图上的每个结点至多只有一个父结点，所以在转换成无向图的时候可以直接将箭头的方向去掉即可。

<img src="figures/08-22-inference-on-undirected-chain.png" style="zoom:40%" align=center />

​    根据无向图的因子化公式，上面的无向链图的联合概率分布为
$$
p(\mathbf{x}) = \frac{1}{Z}\psi_{1,2}(x_1, x_2)\psi_{2,3}(x_2, x_3)\cdots\psi_{N, N-1}(x_N, x_{N-1}).
$$
其中$\mathbf{x}$表示所有结点构成的随机变量矢量。为了简化表述，并给出更直观的描述，我们假设图中每个结点可以取K个状态，所以每个势函数$\psi_{n, n-1}(x_n, x_{n-1})$有$K^2$个参数，总共有$N-1$个势函数，所以联合概率分布有$(N-1)K^2$个参数。

​    对某个结点$x_n$的推断问题，即求解marginal概率
$$
p(x_n) = \sum_{x_1}\cdots\sum_{x_{n-1}}\sum_{x_{n+1}}\cdots\sum_{x_N}p(\mathbf{x}).
$$
简单的分析，求解$x_n$的推断问题，需要首先求解联合概率分布$p(\mathbf{x})$，涉及到$\mathbf{x}$的N个随机变量分量，每个随机变量有$K$个状态，所以$\mathbf{x}$总共有$K^N$个值。因此求解$x_n$的推断问题需要存储和计算的量与链图的长度$N$呈指数关系。

​    我们可以进一步观察上面无向链图的规律，对推断问题进行优化，减少计算量。

​    分析联合概率分布$p(\mathbf{x})$中的每个势函数实际上只涉及到相邻的两个结点。把联合概率分布公式代入$p(x_n)$中，
$$
p(x_n) = \frac{1}{Z}\sum_{x_1}\cdots\sum_{x_{n-1}}\sum_{x_{n+1}}\cdots\sum_{x_N}\psi_{1,2}(x_1, x_2)\psi_{2,3}(x_2, x_3)\cdots\psi_{N, N-1}(x_N, x_{N-1}).
$$
整个联合概率分布中对$x_N$的求和只与最后一个势函数$\psi_{N,N-1}(x_N, x_{N-1})$有关，所以可以讲求和号移到最后一个势函数之前，成为$\sum_{x_N}\psi_{N,N-1}(x_N, x_{N-1})$，这一项是$x_{N-1}$的函数，同样可以把对$x_{N-1}$的求和号移到它之前。依此类推，可以把从$x_{n+1}$到$x_N$的势函数的求和号分别放在给自势函数之前。同理，从$x_1$开始到$x_{n-1}$的势函数的求和号也放在给自势函数之前，
$$
\begin{align}
p(x_n)&=\frac{1}{Z}\\
&\times\left[\sum_{x_{n-1}}\psi_{n-1,n}(x_{n-1},x_n)\cdots\left[\sum_{x_2}\psi_{2,3}(x_2,x_3)\left[\sum_{x_1}\psi_{1,2}(x_1,x_2)\right]\right]\right]\\
&\times\left[\sum_{x_{n+1}}\psi_{n,n+1}(x_n,x_{n+1})\cdots\left[\sum_{x_N}\psi_{N-1,N}(x_{N-1},x_N)\right]\right].
\end{align}
$$
上式中从$x_1$到$x_{n-1}$的求和是$x_n$的函数，记作$\mu_\alpha(x_n)$；从$x_N$到$x_{n+1}$的求和也是$x_n$的函数，记作$\mu_\beta(x_n)$。因此，$x_n$的marginal概率可以写成
$$
p(x_n) = \frac{1}{Z}\mu_\alpha(x_n)\mu_\beta(x_n).
$$
求解上式，需要计算三个量$\mu_\alpha(x_n)$, $\mu_\beta(x_n)$和配分函数$Z$。

​    用迭代方法分析$\mu_\alpha(x_n)$,
$$
\begin{align}
\mu_\alpha(x_n) &= \sum_{x_{n-1}}\psi_{n-1,n}(x_{n-1}, x_n)\left[\sum_{x_{n-2}}\psi_{n-2, n-1}(x_{n-2}, x_{n-1})\cdots\right]\\
&=\sum_{x_{n-1}}\psi_{n-1, n}(x_{n-1}, x_n)\mu_\alpha(x_{n-1}).
\end{align}
$$
因此可以首先计算第一项
$$
\mu_\alpha(x_2)=\sum_{x_1}\psi_{1,2}(x_1,x_2),
$$
然后迭代的计算$\mu_\alpha(x_3), \cdots, \mu_\alpha(x_n).$

​    使用相同的迭代方法，可以从$\mu_\beta(x_{N-1})$出发，依次计算$\mu_\beta(x_{N-2}), \cdots, \mu_\beta(x_n)$.

​    $\mu_\alpha(x_n)​$和$\mu_\beta(x_n)​$的迭代计算过程，可以表示成信息从无向链图的两端$x_1​$和$x_N​$分别传递到$x_n​$，如下图。下图称为马尔科夫链（Markov chain），对应的信息传递迭代方程是马尔科夫过程的Chapman-Kolmogorov方程。

<img src="figures/08-23-messages-transfer-on-undirected-chain.png" style="zoom:40%" align=center />

​    配分函数$Z$可以很容易的通过对marginal概率$p(x_n)$的所有$K$个状态求和得到。

​    如果需要计算无向链图中每个结点的先验概率$p(x_i), \forall i\in \{1, 2, \cdots, N\}​$，则只需分别计算信息从$x_1​$向$x_N​$传递的函数$\mu_\alpha(x_2), \mu_\alpha(x_3), \cdots, \mu_\alpha(x_N)​$，以及信息从$x_N​$向$x_1​$传递的函数$\mu_\beta(x_{N-1}), \mu_\beta(x_{N-2}), \cdots, \mu_\beta(x_1)​$，然后可以根据求解先验概率的结点$x_i​$，挑选出需要的函数进行组合。

​    上面的方法可以计算无向链图中的某一个结点对应随机变量$x_n$的推断问题，即求解$p({x_n})$.下面我们考虑两个推广情况：

1）假设无向链图中某个结点$x_m$为观测量，然后求解另外一个结点$x_n$的条件概率，即求解$p(x_n|x_m)$。此时只需要在上面的迭代公式中加入一项$I(x_m, \hat{x}_m)$，其中$\hat{x}_m$是结点$x_m$的观测值；当$x_m=\hat{x}_m$时，$I=1$，其他情况$I=0$.

2）求解无向链图中两个相邻的结点$x_n​$和$x_{n+1}​$的联合推断问题，即求解$p(x_n, x_{n+1})​$。此时需要将上面迭代公式中对$x_n​$和$x_{n+1}​$的求和号去掉，仍然利用信息从链图的两个端点向中间传递的方法计算得到
$$
p(x_n, x_{n+1}) = \frac{1}{Z}\mu_\alpha(x_n)\psi_{n, n+1}(x_n, x_{n+1})\mu_\beta(x_{n+1}).
$$

- **树图（Trees）**

​    链图是一种结构简单的图模型，下面我们将链图推广到树图中去，并将前面讲解的链图的信息传递方法推广到树图上的sum-product algorithm.

​    1）无向图中的树图：图中任意一对结点之间有且只有一条路径进行连通，见下图(a)。可见无向图中的树图没有loop。

​    2）有向图中的树图：只能有一个结点没有父结点，称为根结点（root），其他的所有结点都只能又一个父结点，见下图（b）。有向图转换成无向图的方法中，如果有向图中某个结点有多个父结点，那么在转换过程中，moralization会在这些多个父结点之间进行全连通，这样会形成loop，因此有向图中的树图的定义要求有向图中结点的父结点数目不超过1。

​    3）有向图中的多树图（polytree）：有些结点可以有超过一个父结点，但是所有结点之间只能有一条路径（不考虑箭头方向）连通，见下图（c）。多树图转换成无向图后其中含有loop。

​    <img src="figures/08-24-tree-graphs.png" style="zoom:40%" align=center />

- **因子图（Factor graphs）**

​    sum-product algorithm对无向图中的树图、有向图中的树图和多树图都适用，我们可以采用一种更普遍的因子图（fator graph）的形式进行描述。

​    有向图和无向图的联合概率分布都可以写成一系列与结点相关的量的乘积形式。因子图增加了因子结点，表示这些与结点相关的量。因子结点用实心的方块表示，见下图。因子结点表示联合概率分布中相乘的量，用$f_i$表示。因子结点与随机变量结点之间用没有方向的线连接。

<img src="figures/08-25-factor-graph.png" style="zoom:40%" align=center />

​    因子图中联合概率分布表示成
$$
p(\mathbf{x}) = \prod_s f_s(\mathbf{x}_s),
$$
其中$\mathbf{x}_s$表示与因子结点$f_s$相连的随机变量结点的集合。对于有向图的因子图，因子$f_s(\mathbf{x}_s)$是一些局域的条件概率。对于无向图的因子图，因子$f_s(\mathbf{x}_s)$是maximal clique的势函数，归一化因子$1/Z$可以看成是不与任何随机变量结点相连的因子结点的因子。根据上面的因子图，写出联合概率分布为
$$
p(\mathbf{x})=f_a(x_1,x_2)f_b(x_1, x_2)f_c(x_2,x_3)f_d(x_3).
$$
其中$f_a(x_1, x_2)$与$f_b(x_1,x_2)$，$f_c(x_2,x_3)$与$f_d(x_3)$分别可以写成一项，但是分开写可以给出联合概率分布因子化更细节的信息。

1）无向图转换成因子图：

首先将无向图中的随机变量结点保留，然后按照maximal clique添加因子结点，maximal clique中的随机变量因子都与相应的因子结点相连。同一个无向图可以对应于多个因子图，如下图的因子图(b)和(c)都对应于无向图(a)。

<img src="figures/08-26-undirected-graph-to-factor-graph.png" style="zoom:40%" align=center />

(a)的因子化：$p(\mathbf{x})=\frac{1}{Z}\psi(x_1, x_2, x_3)$

(b)的因子化：$p(\mathbf{x})=f(x_1, x_2, x_3)$

(c)的因子化：$p(\mathbf{x})=f_a(x_1, x_2, x_3)f_b(x_2, x_3)$

2）有向图转换成因子图：

首先将有向图中的随机变量结点保留，然后按照条件概率中相关的随机变量的关系添加因子结点，再将这些随机变量与相应的因子结点连接起来。同样的，同一个有向图可以对应于多个因子图，如下图的因子图(b)和(c)都对应于有向图(a)。

<img src="figures/08-27-directed-graph-to-factor-graph.png" style="zoom:40%" align=center />

(a)的因子化：$p(\mathbf{x})=p(x_1)p(x_2)p(x_3|x_1, x_2)$

(b)的因子化：$p(\mathbf{x})=f(x_1, x_2, x_3)$

(c)的因子化：$p(\mathbf{x})=f_a(x_1)f_b(x_2)f_c(x_1, x_2, x_3)$

3）有向多树图转换成因子图：

有向多树图(a)转换成无向图(b)时，因为有些结点有多个父结点，所以在moralization的过程中会形成loop。但是有向多树图转换成因子图(c)时，让可以保留树结构，不形成loop。

<img src="figures/08-28-directed-polytree-to-factor-graph.png" style="zoom:40%" align=center />

4）有向循环图转换成因子图：

有向循环图(a)转换成因子图(b)时，可以接触loop结构。

<img src="figures/08-29-directed-loop-graph-to-factor-graph.png" style="zoom:40%" align=center />

(a)的因子化：$p(\mathbf{x})=p(x_1)p_(x_2|x_1)p(x_3|x_1, x_2)$

(b)的因子化：$p(\mathbf{x})=f(x_1, x_2, x_3)$

5）无向循环图转换成因子图：



<img src="figures/08-30-undirected-loop-graph-to-factor-graph.png" style="zoom:40%" align=center />

(a)的因子化：$p(\mathbf{x})=\frac{1}{Z}\psi(x_1, x_2, x_3)$

(b)的因子化：$p(\mathbf{x})=f(x_1, x_2 x_3)$

(c)的因子化：$p(\mathbf{x})=f_a(x_1, x_2)f_b(x_1, x_3)f_c(x_2, x_3)$



- **The sum-product algorithm**

​    下面我们用因子图来推导tree-structured graphs的一种精确推断算法(exact inference algorithm)，称为sum-product algorithm，可以用来计算因子图中一个或多个结点的marginal概率。

​    同样的，为了表述方便，我们假设因子图中的随机变量都是离散的，而且每个随机变量都有$K$个状态。

​    另外，我们假设初始的图模型是无向树图、有向树图或有向多树图，这样可以保证对应的因子图也具有树形结构。我们首先把原始的无向树图、有向树图或有向多树图转换成因子图，这样就可以用相同的方法进行因子图的处理。

​    在树形结构的因子图中，我们可以选择某一个随机变量结点$x$，然后考虑与$x$相连的因子结点及其与之相连的子树图，见下图。

<img src="figures/08-31-sum-product-algorithm-1.png" style="zoom:40%" align=center />

由此可以把因子图的联合概率分布写成
$$
p(\mathbf{x})=\prod_{s\in\text{ne}(x)}F_s(x, \mathbf{X}s),
$$
其中$\text{ne}(x)$表示与随机变量结点$x$相连的所有因子结点的集合，$F_s$是与因子结点$s$相连的子树的因子化，$\mathbf{X}_s$是与因子结点$s$相连的子树上的随机变量结点的集合。

​    对于推断问题，求解某个结点$x$的marginal概率
$$
p(x)=\sum_{\mathbf{x}\backslash x}p(\mathbf{x}),
$$
其中$\mathbf{x}\backslash x$表示随机变量集合$\mathbf{x}$中除$x$之外的其他随机变量。

​    将因子图表示的联合概率分布$p(\mathbf{x})$代入$p(x)$的marginal概率公式中，并调整求和与求乘积的顺序（求和是对除$x$之外的所有其他随机变量结点进行，求乘积是对与$x$相连的所有因子结点进行，所以二者顺序可以交换），得到
$$
\begin{align}
p(x)&=\sum_{\mathbf{x}\backslash x}\left[\prod_{s\in\text{ne}(x)}F_s(x,\mathbf{X}_s)\right]\\
&=\prod_{s\in\text{ne}(x)}\left[\sum_{\mathbf{x}_s}F_s(x, \mathbf{X}_s)\right]\\
&=\prod_{s\in\text{ne}(x)}\mu_{f_s\rightarrow x}(x).
\end{align}
$$
其中$\mu_{f_s\rightarrow x}(x)=\sum_{\mathbf{X}_s}F_s(x,\mathbf{X}_s)​$表示信息从因子结点$f_s​$传递到随机变量结点$x​$，所以$x​$的marginal概率$p(x)​$可以理解成是从与$x​$相连的所有因子结点流入$x​$的信息。

​    接下来进一步分析与因子结点$s$相连的子图的因子化，
$$
F_s(x, \mathbf{X}_s) = f_s(x, x_1,\cdots,x_M)G_1(x_1, \mathbf{X}_{s1})\cdots G_M(x_M, \mathbf{X}_{sM}),
$$
$(x, x_1, \cdots, x_M)$是与因子结点$s$相连的随机变量结点，子图的因子化等于$s$结点的因子$f_s$和与$s$相连的其他随机变量结点$\mathbf{X}_s=(x_1, \cdots, m_M)$相连的子图的因子化$G_1, \cdots, G_M$的乘积。将$F_s(x, \mathbf{X}_s)$代入$\mu_{f_s\rightarrow x}(x)$中，得到
$$
\begin{align}
\mu_{f_s\rightarrow x}(x) &= \sum_{\mathbf{X}_s}F_s(x, \mathbf{X}_s)\\
&= \sum_{x_1}\cdots\sum_{x_M}f_s(x, x_1, \cdots, x_M)\prod_{m\in\text{ne}(f_s)\backslash x}\left[\sum_{\mathbf{X}_{xm}}G_m(x_m, \mathbf{X}_{sm})\right]\\
&= \sum_{x_1}\cdots\sum_{x_M}f_s(x, x_1, \cdots, x_M)\prod_{m\in\text{ne}(f_s)\backslash x}\mu_{x_m\rightarrow f_s}(x_m). 
\end{align}
$$
其中$\text{ne}(f_s)​$表示与因子结点$f_s​$相连的所有随机变量结点的集合，$\mu_{x_m\rightarrow f_s}(x_m)=\sum_{\mathbf{X}_{sm}}G_m(x_m, \mathbf{X}_{sm})​$表示从随机变量结点$x_m​$向因子结点$f_s​$传递的信息。注意：$X_s​$是因子结点$s​$对应的子图上的所有随机变量，$X_{xm}​$表示随机变量结点$x_m​$对应的子图上的所有随机变量，所以$X_s = \{x_1, \cdots, x_M\} \bigcup X_{xm}​$。上式可以理解成，由因子结点$s​$向随机变量结点$x​$传递的信息等于：因子结点$s​$的因子$f_s​$与随机变量结点$(x_1, \cdots, x_M)​$传递到因子结点$s​$的信息 $\mu_{x_m\rightarrow f_s(x_m)}​$的乘积，然后对$(x_1, \cdots, x_M)​$求和。直观的表示见下图。

<img src="figures/08-32-sum-product-algorithm-2.png" style="zoom:40%" align=center />

接下来再分析从随机变量结点$x_m​$向因子结点$f_s​$传递的信息 $\mu_{x_m\rightarrow f_s}(x_m)​$. 由下图可以看出与随机变量结点$x_m​$相连的子图$G_m(x_m, X_{sm})​$的因子化等于与之相连的因子结点$\{f_1, \cdots, f_L\}​$的子图的因子化的乘积，即
$$
G_m(x_m, X_{sm}) = \prod_{l\in\text{ne}(x_m)\backslash f_s} F_l(x_m, X_{ml}),
$$
其中$X_{ml}$表示因子结点$f_l$的子图上的所有随机变量结点的集合，$F_l(x_m, X_{ml})$是因子结点$f_l$的子图的因子化。

<img src="figures/08-33-sum-product-algorithm-3.png" style="zoom:40%" align=center />

将$G_m(x_m, X_{ml})$代入$\mu_{x_m\rightarrow f_s}(x_m)$，得到
$$
\begin{align}
\mu_{x_m\rightarrow f_s}(x_m) &= \sum_{X_{sm}}\prod_{l\in\text{ne}(x_m)\backslash f_s} F_l(x_m, X_{ml})\\
&= \prod_{l\in\text{ne}(x_m)\backslash f_s} \left[\sum_{X_{ml}} F_l (x_m, X_{ml}\right]\\
&=  \prod_{l\in\text{ne}(x_m)\backslash f_s}  \mu_{f_l\rightarrow x_m}(x_m),
\end{align}
$$
所以从随机变量结点$x_m​$向因子结点$f_s​$传递的信息等于：从相应的因子结点$\{f_1, \cdots, f_L\}​$流入随机变量结点$x_m​$的信息相乘。



比较从随机变量结点$x​$向因子结点$s​$传递的信息 $\mu_{x\rightarrow f_s}(x)​$和从因子结点$s​$向随机变量结点$x​$传递的信息 $\mu_{f_s\rightarrow x}(x)​$，可以发现：

1）从随机变量结点$x​$向因子结点$s​$传递的信息 $\mu_{x\rightarrow f_s}(x)​$是所有除$s​$外其他与$x​$相连的因子结点的信息的乘积；

2）从因子结点$s​$向随机变量结点$x​$传递的信息 $\mu_{f_s\rightarrow x}(x)​$是因子结点$s​$本身的因子$f_s​$和所有除$x​$外与$s​$相连的随机变量结点的信息的乘积。



我们沿着树图一直追溯至最终的结点（称为叶结点），如果叶结点是随机变量结点$x​$，那么它向与之相连的因子结点$f​$传递的信息为
$$
\mu_{x\rightarrow f}(x) =1.
$$
如果叶结点是因子结点$f​$，那么它向与之相连的随机变量结点$x​$传递的信息为
$$
\mu_{f\rightarrow x}(x) = f(x).
$$
具体的图表示为

<img src="figures/08-34-leaf-node.png" style="zoom:40%" align=center />

**小结：求解树形结构中某个随机变量$x$的推断问题**

即求解$p(x)$。

1）首先将$x​$作为根结点，从叶结点出发，根据叶结点的类型利用公式$\mu_{x\rightarrow f}(x) = 1​$或$\mu_{f\rightarrow x}(x)= f(x)​$计算叶结点流出的信息；

2）利用信息传递公式，一直迭代的计算到根结点：
$$
\begin{align}
\text{因子结点向随机变量结点传递message:}\quad &\mu_{f_s\rightarrow x}(x) =  \sum_{x_1}\cdots\sum_{x_M}f_s(x, x_1, \cdots, x_M)\prod_{m\in\text{ne}(f_s)\backslash x}\mu_{x_m\rightarrow f_s}(x_m) \\
\text{随机变量结点向因子结点传递message:}\quad & \mu_{x_m\rightarrow f_s}(x_m) =   \prod_{l\in\text{ne}(x_m)\backslash f_s}  \mu_{f_l\rightarrow x_m}(x_m),
\end{align}
$$
==计算树形图中每个随机变量的marginal概率==，需要计算$p(x), \hspace{0.1cm} \forall x\text{是图中任意随机变量}$. 我们只需要计算每个结点处信息向两个方向传递的函数，就可以根据具体的随机变量结点$x$的位置组合出$p(x)$，因此计算每个随机变量的marginal概率的计算量是计算一个随机变量的marginal概率的计算量的2倍。

==计算树形图中与某个因子结点$s​$相连的所有随机变量的联合概率分布$p(\mathbf{x}_s)​$==。与计算一个随机变量的marginal概率$p(x)​$的情况不同的是，这里不需要对与$s​$相连的随机变量求和。联合概率分布$p(\mathbf{x}_s)​$是结点$s​$的因子，以及从各个与$s​$相连的随机变量结点流入$s​$的信息的乘积，
$$
p(\mathbf{x}_s) = f_s(\mathbf{x}_s)\prod_{i\in\text{ne}(f_s)}\mu_{x_i\rightarrow f_s}(x_i).
$$
==计算树形图中的后验概率==。假设树形图中的随机变量$\mathbf{x}$分为两部分，一部分为隐变量$\mathbf{h}$，另一部分为观测量$\mathbf{v}$，观测量的观测值为$\hat{\mathbf{v}}$，则计算后验概率时只需要在联合概率分布$p(\mathbf{x})$前加上$\prod_i I(v_i, \hat{v}_i)$，当$v=\hat{v}_i$时，$I(v_i, \hat{v}_i) = 1$，否则为0，即
$$
p(\mathbf{h}|\mathbf{v}=\hat{\mathbf{v}})=\prod_{i} p(\mathbf{x})I(v_i, \hat{\mathbf{v}}_i).
$$
==因子图的归一化问题==：如果因子图是从有向图转换过来的，因为因子$f$可以直接用有向图中的条件概率类比过来，所以天然的满足归一化条件；如果因子图是从无向图转换过来的，因子$f$不满足归一化条件，假设用sum-product algorithm计算的任一个随机变量$x$的未归一化marginal概率$\tilde{p}(x)$计算，则归一化系数$Z=\sum_x \tilde{p}(x)$。

**例：**一个含有4个随机变量结点的因子图的计算

下图中含有4个随机变量结点$x_1, x_2, x_3, x_4$和3个因子结点$f_a, f_b, f_c$，未归一化的联合概率分布为
$$
\tilde{p}(\mathbf{x}) = f_a(x_1, x_2)f_b(x_2, x_3)f_c(x_2, x_4).
$$
<img src="figures/08-36-sum-product-algorithm-example-4-nodes-1.png" style="zoom:40%" align=center />

首先选取$x_3​$作为根结点，则$x_1,x_4​$位叶结点，图中总共有6个信息传递函数（见下图(a)）：
$$
\begin{align}
\mu_{x_1\rightarrow f_a} (x_1) &= 1\\
\mu_{f_a\rightarrow x_2}(x_2) &= \sum_{x_1}f_a(x_1, x_2)\\
\mu_{x_4\rightarrow f_c}(x_4) &= 1\\
\mu_{f_c\rightarrow x_2}(x_2) &= \sum_{x_4}f_c(x_2, x_4)\\
\mu_{x_2\rightarrow f_b}(x_2) &= \mu_{f_a\rightarrow x_2}(x_2)\mu_{f_c\rightarrow x_2}(x_2)\\
\mu_{f_b\rightarrow x_3}(x_3) &= \sum_{x_2}f_b(x_2, x_3)\mu_{x_2\rightarrow f_b}(x_2)
\end{align}
$$
我们另外计算从根结点$x_3​$到叶结点$x_1, x_4​$的信息传递函数（见下图(b)）：
$$
\begin{align}
\mu_{x_3\rightarrow f_b}(x_3) &= 1\\
\mu_{f_b\rightarrow x_2}(x_2) &= \sum_{x_3}f_b(x_2, x_3)\\
\mu_{x_2\rightarrow f_a}(x_2) & = \mu_{f_b\rightarrow x_2}(x_2)\mu_{f_c\rightarrow x_2}(x_2)\\
\mu_{f_a\rightarrow x_1}(x_1) &= \sum_{x_2}f_a(x_1, x_2)\mu_{x_2\rightarrow f_a}(x_2)\\
\mu_{x_2\rightarrow f_c}(x_2) &= \mu_{f_a\rightarrow x_2}(x_2)\mu_{f_b\rightarrow x_2}(x_2)\\
\mu_{f_b\rightarrow x_4}(x_4) &= \sum_{x_2}f_c(x_2, x_4)\mu_{x_2\rightarrow f_c}(x_2)
\end{align}
$$
<img src="figures/08-37-sum-product-algorithm-example-4-nodes-2.png" style="zoom:40%" align=center />

如果我们要计算$x_2​$的marginal概率$p(x_2)​$，可以将需要的信息传递函数组合起来，首先计算未归一化概率
$$
\begin{align}
\tilde{p}(x_2)&=\mu_{f_a\rightarrow x_2}(x_2)\mu_{f_b\rightarrow x_2}(x_2)\mu_{f_c\rightarrow x_2}(x_2)\\
&= \left[\sum_{x_1}f_a(x_1, x_2)\right]\left[\sum_{x_3}f_b(x_2, x_3)\right]\left[\sum_{x_4}f_c(x_2, x_4)\right]\\
&= \sum_{x_1}\sum_{x_3}\sum_{x_4}f_a(x_1, x_2)f_b(x_2, x_3)f_c(x_2, x_4)\\
&= \sum_{x_1}\sum_{x_3}\sum_{x_4}\tilde{p}(\mathbf{x})
\end{align}
$$

- **The max-sum algorithm**

在统计推断中，我们经常会遇到两个问题：

1）使联合概率分布$p(\mathbf{x})$取最大值的随机变量的取值是什么，即求解 $$\mathbf{x}^\max=\mathop{\arg\max}_\mathbf{x} p(\mathbf{x})$$,

2）求解联合概率分布的最大值 $$p(\mathbf{x}^\max)=\max_\mathbf{x} p(\mathbf{x})$$.

如果使用前面讲解的sum-product algorithm，求得每个随机变量$x_i$的marginal概率$p(x_i)$，然后求$$x_i^\max = \mathop{\arg\max}_{x_i}p(x_i)$$，那么求得的结果只是$p(x_i)$的局域极大化，不是$p(\mathbf{x})$的全局最大化。

下面我们讲解max-sum algorithm，能够求解联合概率分布$p(\mathbf{x})$的全局最大化。max-sum algorithm动态规划（dynamic programming）在图模型中的一个应用。

==我们首先讨论第二个问题：求解树形图中联合概率分布的最大值==。

先以前面讨论的链图作为例子，
$$
\begin{align}
\max_{\mathbf{x}}p(\mathbf{x})&=\frac{1}{Z}\max_{\mathbf{x}}\left[\psi_{1,2}(x_1,x_2)\cdots\psi_{N-1,N}(x_{N-1},x_N)\right]\\
&=\frac{1}{Z}\max_{x_1}\max_{x_2}\cdots\max_{x_N}\left[\psi_{1,2}(x_1,x_2)\cdots\psi_{N-1,N}(x_{N-1},x_N)\right]\\
&=\frac{1}{Z}\max_{x_1}\left[\max_{x_2}\psi_{x_1,x_2}(x_1,x_2)\left[\cdots\max_{x_N}\psi_{N-1,N}(x_{N-1},x_N)\right]\right].
\end{align}
$$
上式中对所有$\mathbf{x}$求最大值的可以拆分成对每个结点$x_i$求最大值，因为$p(\mathbf{x})$的特殊结构，可以写成一些局域行的因子$\psi_{i, i+1}(x_i,x_{i+1})$的乘积，求最大值可以只针对与之相关的项进行，所以可以调整乘积与$\max$de 顺序。

然后再将链图中求联合概率最大化的思想推广到树形图中。sum-product algorithm巧妙的根据树形图的特征调整联合概率分布中各局域化的因子的乘积和对除根结点外的其他随机变量求和的顺序，与之类似，此处也可以调整乘积和求最大值$\max​$的顺序。把对所有随机变量的最大值拆解组成对每个随机变量求最大值，然后从叶结点开始，沿着信息向根结点传递的方向依次求局域化的最大值。这种方法称为max-product algorithm。

在实际应用中，我们利用对数函数的单调性，对联合概率分布求对数，将因子之间的乘积关系转换成因子对数的求和，求解$$\max_{\mathbf{x}}\ln p(\mathbf{x})$$，这样修改后的方法称为max-sum algorithm。

下面给出==max-sum algorithm的操作步骤==：

1）首先选取一个随机变量结点$x$作为根结点，联合概率分布的最大值为
$$
p^\max=\max_x p(x) = \max\left[\sum_{s\in\text{ne}(x)}\ln\mu_{f_s\rightarrow x}(x)\right],
$$
2）计算叶结点传递出来的信息，
$$
\begin{align}
\ln\mu_{x\rightarrow f}(x) &= 0\\
\ln\mu_{f\rightarrow x}(x) & = \ln f(x),
\end{align}
$$
3）计算从叶结点到根结点的信息传递函数，
$$
\begin{align}
\ln\mu_{f\rightarrow x}(x) &= \max_{x_1,\cdots, x_M}\left[\ln f(x, x_1, \cdots, x_M) + \sum_{m\in\text{ne}(f)\backslash x}\ln\mu_{x_m\rightarrow f}(x_m)\right]\\
\ln\mu_{x\rightarrow f}(x) &= \sum_{l\in\text{ne}(x)\backslash f}\ln\mu_{f_l\rightarrow x}(x).
\end{align}
$$
==接下来我们讨论第一个问题：求解使联合概率分布取最大值的随机变量$\mathbf{x}$==

在求解第一个问题的过程中，按照迭代方法，最后计算根结点$x$时，也可以计算出联合概率分布取最大值时根结点$x$的取值，
$$
x^\max = \mathop{\arg\max}_x\left[\sum_{s\in\text{ne}(x)}\ln\mu_{f_s\rightarrow x}(x)\right].
$$
我们可能会简单推想，从根结点出发，按照信息传递的反方向依次求解使联合概率分布最大化的其他随机变量结点的取值，但是因为这种多变量求极值的问题中可能有多个随机变量组合都对应着联合概率分布的全局最大值，而根据上面公式对每个结点依次求极值的过程可能选取的各个随机变量属于不同的最大随机变量组合，这样求的结果不一定是全局最大值。

下面我们以前面讨论过的链图来解释求解联合概率分布区最大值的随机变量组合(configuration)$\mathbf{x}=\{x_1,\cdots, x_N\}$。

以最后一个随机变量结点$x_N$为根结点，第一个随机变量结点$x_1$为叶结点。

首先计算信息传递函数，
$$
\begin{align}
\ln\mu_{x_n\rightarrow f_{n,n+1}}(x_n) &= \ln\mu_{f_{n-1,n}\rightarrow x_n}(xn)\\
\ln\mu_{f_{n-1,n}\rightarrow x_n}(x_n) &= \max_{x_{n-1}}\left[\ln f_{n-1,n}(x_{n-1}, x_n)+\ln\mu_{x_{n-1}\rightarrow f_{n-1,n}(x_n)}\right],
\end{align}
$$
叶结点的信息传递函数,
$$
\mu_{x_1\rightarrow f_{1,2}}(x_1)=0.
$$
然后计算联合概率分布取最大值时根结点$x_N$的取值，
$$
x_N^\max =\mathop{\arg\max}_{x_N}\left[\ln\mu_{f_{N-1,N}\rightarrow x_N}(x_N)\right].
$$
为了确定根结点之前的结点和根结点都属于同一个随机变量最大化组合中，我们在求根结点之前的结点的最大化时，需要记住后面的结点的最大化。我们通过下面的公式来做这样的存储，
$$
\phi(x_n)=\mathop{\arg\max}_{x_{n-1}}\left[\ln f_{n-1,n}(x_{n-1},x_n) + \ln \mu_{x_{n-1}\rightarrow f_{n-1,n}}(x_n)\right],
$$
最后通过回溯过程(back-tracking)，求得在同一个随机变量最大化组合中的个结点最大化取值，
$$
\begin{align}
x_{n-1}^\max &= \phi(x_n^\max)\\
&= \mathop{\arg\max}_{x_{n-1}}\left[\ln f_{n-1,n}(x_{n-1},x_n^\max) + \ln \mu_{x_{n-1}\rightarrow f_{n-1,n}}(x_n^\max)\right].
\end{align}
$$
注意，这种回溯过程在每次往前求解随机变量最大化取值时，都记住了靠近根结点的随机变量的最大化取值。下图直观的解释了这个过程，图中假设链图中信息从左向右传递，每个随机变量有3个状态，求解最大化随机变量组合时，先从根结点出发向左依次求解。图中两条黑线从右向左的给出了两个最大化随机变量组合。这种图称为格子图（lattice diagram或trellis diagram）。

<img src="figures/08-38-max-sum-algorithm-of-chain.png" style="zoom:40%" align=center />

参考链图的例子，可以推广到求解一般的树形图的联合概率分布最大化的随机变量组合问题。从根结点出发，依次向前求解局域的最大化随机变量取值。例如，某个最大化随机变量取值$x^\max$已求得，与其相连的因子结点为$f$，除$x$外$f$还与随机变量结点$x_1,\cdots,x_M$相连，记住$x^\max$后，利用信息传递函数可回溯求出$x_1^\max,\cdots, x_M^\max$。最终max-sum algorithm可以求得使树形图的联合概率分布的全局最大化的随机变量组合(exact maximizing configuration)。

max-sum algorithm的一个重要的应用是在隐马尔可夫模型(hidden Markov model)中的寻找隐态(hidden states)的概率最高的序列，称为Viterbi algorithm.

如果树形图中有些随机变量是观测量，要求解条件概率的最大化问题可以参照sum-product algorithm，在条件概率中添加单位函数$I(x_i, \hat{x}_i)$。



- **Exact infeence in general graphs**


树形图中的信息传递的思想可以推广到任意的图模型上去，从而对一般的图模型进行精确推断(exact inference)，这种算法称为junction tree algorithm。

下面简单讨论junction tree algorithm的思想：

1）如果原始的图是有向图，则用moralization方法将有向图转换成无向图；如果原始的图是无向图则不用转换。

2）将无向图三角化(triangulated)。如果无向图中有4个或更多结点组成的无弦环(chord-less cycle)，则通过添加额外的边，将无弦环切割成不含无弦环的图。例如下图中$A-C-B-D-A$构成一个无弦环，可以在$A$和$B$之间或$C$和$D$之间添加边，消除无弦环。

<img src="figures/08-39-loop-grapha.png" style="zoom:40%" align=center />

3）利用三角化后的无向图构造聚合树(join tree)。join tree的结点对应于三角化后的无向图的maximal cliques，它的边连接有共同结点的clique。选择哪些cliques对进行连接很重要，需要保证得到最大生成树(maximal spanning tree)。具体操作方法是在所有连接cliques的可能的树当中，选择权值(weight)最大的树，这样的到的树就是join tree。边的权值是指两个相连的cliques共同拥有的结点数目，树的权值是指所有边的权值之和。由于进行了三角化，所以join tree满足running intersection property，即如果一个随机变量被两个cliques包含，那么他一定也被连接这两个cliques的路径上的所有clique所包含。这个性质确保了对随机变量的推断是相容的(consistent)。

4）最后类比sum-product algorithm，经过两个信息传递步骤，可以计算junction tree的marginal概率和条件概率。

junction tree algorithm看起来很复杂，但是背后的思想很简单，主要是==使图的因子化形式（即联合概率分布的形式）具有一些局域化的性质，能够在求marginal时能保证乘积与求和的顺序交换==。

junction tree algorithm的计算量由join tree中的最大clique所包含的随机变量个数决定，这个数称为join tree的树宽(treewidth)。如果是离散随机变量，junction tree algorithm的计算量与树宽呈指数关系。在实际应用当中，一般将树宽定义成join tree中最大的clique所包含的随机变量个数减去1，因为这样将最小的树的树宽取作1。

一般从原始的树出发，可以构造出多种join trees，我们取其中maximal cliques中含有随机变量最少的一个定义成树宽。如果树宽度太大，计算量很大，junction tree algorithm也就变得不实用。




- **Loopy belief propagation（循环置信传播）**


有些实际问题使用精确推断(exact inference)不可行，有些情况可以使用近似推断(approximate inference)来得到可以接受的推断结果。其中一个重要的分支是变分法(variational methods)，将在第10章讨论。另外一种与确定论相对的方法是取样法(sampling methods)，或者称为Monte Carlo methods，将在第11章讨论。

这里我们讨论一种含有环的图结构的近似推断方法，称为循环置信传播(loopy belief propagation)，它是简单的把sum-product algorithm用到含有环的图结构中去，这样不能保证能得到好的结果。

因为在含有环的图结构中，信息可以在环中循环传播，所以我们需要定义新的信息传递方案(message passing schedule)。假设在任何结点的给定连接和给定方向上，一次只能传递一条信息。一个结点从某个方向收到一条新的信息后，会将它保存的从该方向收到的前一条信息覆盖。如果图中有环形结构，那么其上保存的信息会不断被更新。我们将结点向任何方向传递的信息初始化为单位函数。这样每个结点就可以开始向外发送信息。每次可以同时向各个方向发送信息的情况称为flooding schedule，每次只发送一条信息的情况称为serial schedule。

对于结点（随机变量结点或因子结点）$a$和$b$，如果$a$从其他任何结点接收到信息，准备发送给$b$，那么称为$a$相对$b$的信息挂起(pending message)。

对于树形结构的图，由于不存在环形结构，结点上的信息传出后，该结点的信息就传输完了，因此没有挂起信息(pending message)，得到的是精确推断结果。但是对于有环形结构的图，因为挂起信息一直存在，所以信息的传递永远不会结束。实际应用的很多情况中，信息在环形结构传输若干和循环后会收敛。这样得到的是近似的局域推断结果。




- **Learning the graph structure**


前面讨论的都是图模型的结构已知且固定的情况，还可以讨论根据数据学习图的结构。

定义一个图的结构空间，即模型空间$\mathcal{M}$。根据贝叶斯公式，如果已知模型空间的先验概率$p(m), m\in \mathcal{M}$，观测数据集为$\mathcal{D}$，那么
$$
p(m|\mathcal{D}) \propto p(m)p(\mathcal{D}|m),
$$
其中模型证据(model evidence) $p(\mathcal{D}|m)$相当于提供了给每个模型打分的评判标准。

实际应用当中，学习图结构会有很多难以实施的困难。模型空间$\mathcal{M}$本身不太好定义，而且模型的数量是与结点数呈指数关系的，有时要靠直觉筛选。另外计算$p(\mathcal{D}|m)$需要对模型空间计算隐变量的marginal，需要计算很多模型，计算量非常大。

