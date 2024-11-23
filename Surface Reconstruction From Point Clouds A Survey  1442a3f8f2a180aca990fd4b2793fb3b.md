# Surface Reconstruction From Point Clouds: A Survey and a Benchmark

Select: Survey

# Introduction

**Surface reconstruction** từ point clouds là một bài toán lâu đời trong lĩnh vực computer vision và computer graphics. Point cloud là tập hợp các điểm rời rạc `discrete` trong không gian 3D được thu thập từ các cảm biến như **multi-view stereo**, **structured light**, hoặc **time-of-flight (ToF)**. Tuy nhiên, bài toán này gặp nhiều thử thách `challenges` do:

- *Ill-posed problem*: Với mỗi point cloud, có vô số [**Continuous Surface**](https://www.notion.so/Continuous-Surface-1462a3f8f2a180189662e5f1362945b4?pvs=21)  có thể phù hợp.
- *Imperfections*: Dữ liệu Point Cloud thường có nhiễu (noise), phân bố không đồng đều (non-uniform distribution), hoặc chứa các outliers.

**Mục tiêu của bài báo**:

- Đánh giá các phương pháp surface reconstruction hiện có, bao gồm cả phương pháp cổ điển `classical` và hiện đại `deep learning`.
- Xây dựng một bộ dataset **benchmark** toàn diện, bao gồm dữ liệu tổng hợp `synthetic` và dữ liệu thực tế `real-scanned`, cụ thể `object-` và `scene-level` surfaces.
- Phân tích chi tiết ưu, nhược điểm của các phương pháp hiện có thông qua thực nghiệm.

**Đánh giá tác giả:**

- Các vấn đề về `misaligment`, `missing points`, `outliers` ít được quan tâm và chưa có lời giải
- Các phương pháp deep Learning thể hiện tốt và tiềm năng cho surface modeling và reconstruction. Tuy nhiên, nghiên cứu tác giả đề xuất các phương pháp hiện đại không generalization tốt trong việc tái tạo các bề mặt phức tạp. Nhiều phương pháp classical tốt hơn cả về mặt `generlization` và `robustness`
    
    <aside>
    💡
    
    It is surprising that some classical methods perform even better in terms of both generalization and robustness.
    
    </aside>
    
- Sử dụng pháp tuyến  [**Normal**](https://www.notion.so/Normal-1462a3f8f2a180eb82a1e9cf965039f7?pvs=21) là quan trọng trong task surface reconstruction từ các `raw`, `observed` point clouds. Ngay cả khi normal được xác định kém chính xác, kết quả interior/exterior bề mặt vẫn có thể xác định tốt trong không gian 3D.
    
    <aside>
    💡
    
    The use of surface normals is a key to the success of surface reconstruction from raw, observed point clouds, even when the normals are estimated less accurately
    
    </aside>
    
- Có sự không nhất quán giữa cách đánh giá metrics và trong nhiều trường hợp, kết quả tốt không có nghĩa là visualization được tốt. Tác giả nhận định cần có nhiều phương pháp đánh giá chuẩn mực hơn trong lĩnh vực này.

**Details:**

- [**Normal**](https://www.notion.so/Normal-1462a3f8f2a180eb82a1e9cf965039f7?pvs=21)
- [**Continuous Surface**](https://www.notion.so/Continuous-Surface-1462a3f8f2a180189662e5f1362945b4?pvs=21)
- [**Point Cloud, Polygon meshes, Quantized Volumes ?**](https://www.notion.so/Point-Cloud-Polygon-meshes-Quantized-Volumes-1452a3f8f2a1805f93c6eaf87ebf7e4c?pvs=21)
- [**Multi-view Stereo, Structured Light, Time-of-flight ?**](https://www.notion.so/Multi-view-Stereo-Structured-Light-Time-of-flight-1452a3f8f2a180f3b365fcce22112522?pvs=21)
- [**Ill-posed**](https://www.notion.so/Ill-posed-1442a3f8f2a1809ea44dfc579c47ef7e?pvs=21)
- [**Surface Reconstruction Challenge ?**](https://www.notion.so/Surface-Reconstruction-Challenge-1452a3f8f2a1801787eaee4c6a53fa34?pvs=21)
- [**Dataset**](https://www.notion.so/Dataset-1452a3f8f2a18007b376d1ccbb94ed15?pvs=21)

---

# Problem Statement

Cho một tập hợp điểm rời rạc `discrete point` $P = \{p \in \mathbb{R}^3\}$, đại diện cho các điểm trên bề mặt được scan bởi cảm biến 3D. Nhiệm vụ là tìm một **[**Continuous Surface**](https://www.notion.so/Continuous-Surface-1462a3f8f2a180189662e5f1362945b4?pvs=21) $S^*$** sao cho:

1. **Reconstruct surface** $S^*$ khớp với tập dữ liệu input $P$.
2. $S^*$ thỏa mãn các ràng buộc về hình học `geometry-aware constraints`. (e.g., [Smooth Surface](https://www.notion.so/Smooth-Surface-1462a3f8f2a1804b9930e23513f1fcad?pvs=21))

Tuy nhiên, bài toán này là **`ill-posed`**, vì có vô số bề mặt $S^*$ có thể phù hợp với $P$. Để khắc phục, cần áp dụng regularization để tìm surface $S$ thoả mãn cả **khớp dữ liệu** và **đặc tính hình học.**

Bài toán được biểu diễn bằng một công thức tối ưu hoá tổng quát `regularized optimization`:

$$
\min_{S} \mathcal{L}(S; P) + \lambda \mathcal{R}(S)
$$

Trong đó:

- $\mathcal{L}(S; P)$: Hàm **Loss** ([**Data Fidelity**](https://www.notion.so/Data-Fidelity-1442a3f8f2a180bcbfdef563f8d1ceb7?pvs=21)), đo độ sai lệch giữa surface $S$ và dữ liệu $P$.
- $\mathcal{R}(S)$: **Regularizer**, thêm ràng buộc để kiểm soát hình học của $S$
- $\lambda$: **Scalar penalty**, hệ số điều chỉnh mức độ ưu tiên giữa hai thành phần.

<aside>
⚠️

Lưu ý: công thức trên chỉ dạng `abstract` vì khó có thể xác định $L$ và $R$ trực tiếp trên S

</aside>

Trên thực tế, bài toán thể hiện hai cách biểu diễn: tường minh - **explicit** và ngầm định - **implicit**

## **Explicit Representation**:

Surface $S$ được định nghĩa qua một hàm `explicit representation` $f(x)$ trong không gian 3D:

$$
S = \{f(x) \in \mathbb{R}^3 \mid x \in \Omega^2\}
$$

- $\mathbbΩ^2⊂{R}^2$: miền tham số 2D.
- $f: \Omega^2 \to S$: Hàm ánh xạ tham số từ không gian 2D $\Omega^2$ tới không gian 3D $S$

Công thức tối ưu hóa tổng quát chuyển thành:

$$
\min_{f \in \mathcal{H}_f} \mathcal{L}_{\text{exp}}(f; P) + \lambda \mathcal{R}_{\text{exp}}(f)
$$

- $\mathcal{H}_f$: [**Hypothesis Space**](https://www.notion.so/Hypothesis-Space-1452a3f8f2a18068a3b8d0832f1956fb?pvs=21) chứa các ánh xạ $f$. Ví dụ, $f$ có thể là các đa thức bậc thấp, B-spline, hoặc các mạng neural.
- $\mathcal{R}_{\text{exp}}(f)$: Hàm regularizer, giúp kiểm soát điều kiện surface geometry $f(x)$.

Hàm explicit loss được viết như sau:

$$
\mathcal{L}_{\text{exp}}(f; P) = \frac{1}{n_{p}} \sum_{p \in P} \min_{x \in \Omega^2} \|f(x) - p\|_l
$$

- $n_q$: số lượng point trong tập $P$.
- $\|f(x) - p\|$: khoảng cách Euclidean giữa điểm $f(x)$ trên bề mặt và điểm $p$ trong tập dữ liệu.
- $\min_{x \in \Omega^2}:$ Tìm điểm $f(x)$ gần nhất với $p$ trong không gian 3D.

**Ý nghĩa**: 

- Với mỗi điểm $p \in P$ được “khớp” với ít nhất một điểm $f(x)$ trên bề mặt $S$.

<aside>
⚠️

Thực tế, việc tối ưu trên toàn bộ miền $\Omega^2$ rất phức tạp, nên thường chỉ lấy một tập mẫu $\{x \in \Omega^2\}$
 dựa trên nhiều biến thể `point-set distances` như Chamfer hoặc Hausdorff distances.

</aside>

## **Implicit Representation**:

Trong implicit representation, Surface $S$ được định nghĩa là `zero-level set` của một hàm implicit $F$:

$$
S = \{q \in \mathbb{R}^3 \mid F(q) = 0\}
$$

trong đó $F: \mathbb{R}^3 \to \mathbb{R}$ là hàm implicit định nghĩa bề mặt $S$.

Công thức tối ưu hóa tổng quát chuyển thành:

$$
\min_{F \in \mathcal{H}_F} \mathcal{L}_{\text{imp}}(F; P) + \lambda \mathcal{R}_{\text{imp}}(F)
$$

**Thành phần**:

- $\mathcal{H}_F$: [**Hypothesis Space**](https://www.notion.so/Hypothesis-Space-1452a3f8f2a18068a3b8d0832f1956fb?pvs=21) chứa các hàm $F$, ví dụ: mạng neural hoặc đa thức.
- $\mathcal{R}_{\text{imp}}(f)$: Regularizer, giúp kiểm soát điều kiện surface geometry $f(x)$.

Hàm loss implicit được viết:

$$
\mathcal{L}_{\text{imp}}(F; P) = \frac{1}{n_P} \sum_{p \in P} \|F(p)\|_l
$$

- $F(p)$: giá trị của hàm $F$ tại điểm $p$. Hàm $F$ được thiết kế sao cho $F(p) = 0$  nếu $p$ nằm trên bề mặt, và $F(p) \neq 0$  nếu $p$ cách xa bề mặt. **[**Signed Distance Function (SDF) & Occupancy Field**](https://www.notion.so/Signed-Distance-Function-SDF-Occupancy-Field-1442a3f8f2a18042b4a8f78a96230799?pvs=21)**
- $\|F(p)\|_l$: Hình phạt nếu điểm $p$ không thỏa mãn. Thông thường $l = 1$ hoặc $l = 2$ .
- Ý nghĩa: Công thức này giúp "kéo" các điểm $p$ vào gần bề mặt $S$ bằng cách giảm giá trị $F(p)$.

## **Advanced Implicit Losses**

Bài báo còn đề xuất mở rộng hàm implicit loss để cải thiện chất lượng tái tạo surface, sử dụng công thức:

$$
\mathcal{L}_{\text{imp++}}(F; P) = \alpha_1 \mathbb{E}_{q \in \mathbb{R}^3} \|F(q) - d(q; P)\|^2 + \alpha_2 \mathbb{E}_{q \in \mathbb{R}^3} \|\nabla_q F(q) - n(q; P)\|^2 + ...
$$

Giải thích các thành phần:

1. $d(q; P)$: [**Signed Distance Function (SDF)**](https://www.notion.so/Surface-Reconstruction-From-Point-Clouds-A-Survey-and-a-Benchmark-1442a3f8f2a180aca990fd4b2793fb3b?pvs=21)
    - Khoảng cách có dấu giữa $q$ và bề mặt $S$.
    - $d(q; P) = 0$ khi $q$ nằm trên bề mặt, $d(q; P) > 0$ khi $q$ nằm ngoài bề mặt, và $d(q; P) < 0$ khi nằm trong bề mặt.
2. $\nabla_q F(q)$: [**Gradient**](https://www.notion.so/Surface-Reconstruction-From-Point-Clouds-A-Survey-and-a-Benchmark-1442a3f8f2a180aca990fd4b2793fb3b?pvs=21) của $F$ tại $q$. 
    - Gradient này chỉ hướng pháp tuyến `normal vector` của surface tại $q$.
    - $n(q; P)$: Vector pháp tuyến thực tế tại điểm, được tính toán từ point cloud $P$.
3. Trọng số $\alpha_1, \alpha_2$: điều chỉnh mức độ ưu tiên giữa việc giá trị khoảng cách và vector pháp tuyến.

**Ý nghĩa**:

- Thành phần $\|F(q) - d(q; P)\|^2$: Đảm bảo $F(q)$ mô hình hóa chính xác **Signed Distance Function (SDF)**
- Thành phần $\|\nabla_q F(q) - n(q; P)\|^2$: Cải thiện độ chính xác của bề mặt và hướng pháp tuyến.

![image.png](Surface%20Reconstruction%20From%20Point%20Clouds%20A%20Survey%20%201442a3f8f2a180aca990fd4b2793fb3b/image.png)

## **Kết hợp các hàm loss và regularization**

Do bài toán surface reconstruction là `ill-imposed`, chỉ sử dụng $\mathcal{L}(S; P)$ không đủ để tạo ra một surface hợp lý. Vì vậy, cần thêm **hàm regularizer $\mathcal{R}(S)$** giúp bù đắp hạn chế này:

$$
\min_S \mathcal{L}(S; P) + \lambda \mathcal{R}(S)
$$

Regularizer $\mathcal{R}(S)$ được tác giả nhóm dựa trên các phương pháp như sau:

1. **Triangulation-based prior**
2. **Smoothness prior**
3. **Template-based prior**
4. **Modeling Prior**
5. **Learning-based Prior**
6. **Hybrid Prior**

**Details:**

- [**Explicit & Implicit**](https://www.notion.so/Explicit-Implicit-1442a3f8f2a180759082e4f386e812ad?pvs=21)
- [**Hypothesis Space**](https://www.notion.so/Hypothesis-Space-1452a3f8f2a18068a3b8d0832f1956fb?pvs=21)
- [**Data Fidelity**](https://www.notion.so/Data-Fidelity-1442a3f8f2a180bcbfdef563f8d1ceb7?pvs=21)
- [**Signed Distance Function (SDF) & Occupancy Field**](https://www.notion.so/Signed-Distance-Function-SDF-Occupancy-Field-1442a3f8f2a18042b4a8f78a96230799?pvs=21)
- [**Local Tangent Plane, Normal Vector** ](https://www.notion.so/Local-Tangent-Plane-Normal-Vector-1442a3f8f2a180e9a485de88b59e4fdf?pvs=21)
- [**Gradient**](https://www.notion.so/Gradient-1452a3f8f2a18094954cd5a95913f7ee?pvs=21)

---

# Surface Reconstruction with a Categorization of Geometric Priors

## Triangulation-Based Prior

![image.png](Surface%20Reconstruction%20From%20Point%20Clouds%20A%20Survey%20%201442a3f8f2a180aca990fd4b2793fb3b/image%201.png)

![image.png](Surface%20Reconstruction%20From%20Point%20Clouds%20A%20Survey%20%201442a3f8f2a180aca990fd4b2793fb3b/image%202.png)

[Simple Stereo | Camera Calibration](https://youtu.be/hUVyDabn1Mg?si=qcnRsHgYDfZ4emO4)

Triangulation using Two Cameras

**Triangulation-based prior** là một phương pháp classical trong bài toán *surface reconstruction*. Phương pháp tận dụng giả thiết rằng bề mặt $S$ có thể được biểu diễn bằng một **lưới tam giác (triangular mesh)**, đặc biệt hiệu quả khi bề mặt $S$ có tính chất trơn cục bộ *[**Locally Differentiable**](https://www.notion.so/Locally-Differentiable-1462a3f8f2a18068a642ee6ffb807c5f?pvs=21)  [**Piecewise Linear**](https://www.notion.so/Piecewise-Linear-1462a3f8f2a180fab89dcbfae3d448da?pvs=21) .* Phương pháp này thường áp dụng thuật toán **Delaunay triangulation**, với các bước và công thức toán học cụ thể như sau:

<aside>
💡

A closed surface is continuous and could be locally differentiable up to different orders

</aside>

- Nếu $S$ `khả vi bậc nhất` [**Locally Differentiable up to First Order**](https://www.notion.so/Locally-Differentiable-up-to-First-Order-1462a3f8f2a1809caedbdb095899ba6c?pvs=21), thì ta có thể xấp xỉ $S$
  bằng một tập hợp các **tam giác phẳng (flat triangles)** được ghép lại. Điều này cho phép mô hình hóa bề mặt bằng một **lưới tam giác (triangular mesh)**.
- **Biểu diễn lưới tam giác**: Một lưới tam giác $G_S$ gồm các tam giác $\{T_i\}_{i=1}^{n_G}$, trong đó mỗi tam giác $T$ được xác định bởi ba đỉnh $\{v_1^T, v_2^T, v_3^T\} \subset P$.

**Delaunay triangulation** là một phương pháp triangular mesh phổ biến, với điều kiện:

1. **Empty Sphere Property**: Không có điểm nào trong tập $P$ nằm bên trong mặt cầu ngoại tiếp $\text{CC}(T)$ tam giác $T$.
2. **Tính chất hình học tốt**: Tam giác được tạo ra có góc nhọn lớn nhất có thể, giúp giảm thiểu các tam giác "xấu" (như tam giác dẹt hoặc không đều).

![image.png](Surface%20Reconstruction%20From%20Point%20Clouds%20A%20Survey%20%201442a3f8f2a180aca990fd4b2793fb3b/image%203.png)

[Delaunay Triangulation](https://youtu.be/GctAunEuHt4?si=9eP0ZMIORa1MxxeE)

[Delaunay Triangulation](https://gwlucastrig.github.io/TinfourDocs/DelaunayIntro/index.html)

**Triangular mesh** $G_S$ được xây dựng thỏa mãn:

$$
G_S = \{T_i\}_{i=1}^{n_G}, \quad \text{với } v_1^T, v_2^T, v_3^T \in P, \forall T \in G_S
$$

$$
p \notin \text{CC}(T), \forall p \in P \setminus \{v_1^T, v_2^T, v_3^T\}, \forall T \in G_S
$$

Trong đó:

- $T = \{v_1^T, v_2^T, v_3^T\}$: Một tam giác phẳng được định nghĩa bởi ba điểm $v_1^T, v_2^T, v_3^T$.
- $\text{CC}(T)$: Không gian bên trong mặt cầu ngoại tiếp *[**Circumscribing Sphere of a Triangle**](https://www.notion.so/Circumscribing-Sphere-of-a-Triangle-1462a3f8f2a1801caa51c3fd3bcaf7a0?pvs=21)*  của tam giác $T$.
- **Điều kiện**: Không có điểm nào khác trong $P$ nằm bên trong $CC(T)$.

**Ý nghĩa của công thức**:

- **Empty Sphere Property**: Bảo đảm lưới tam giác được tối ưu về hình học, giảm thiểu tam giác dẹt và tối đa hóa smooth của bề mặt.
- **Đặc điểm phù hợp với dữ liệu**: Tất cả các tam giác được tạo ra đều có đỉnh thuộc tập $P$, bảo đảm reconstruct surface gần với observed $P$.

**Ưu điểm**:

- Phù hợp cho các bề mặt đơn giản.
- Cung cấp cấu trúc lưới có tính chất hình học tốt.
- Linh hoạt với các hình dạng phức tạp trong không gian 2D.

**Nhược điểm**:

- **Không định nghĩa tốt trong không gian 3D**: Khi áp dụng vào không gian 3D, các tam giác có thể không duy trì được tính chất hình học tốt.
- Hầu hết phương pháp tái tạo sử dụng phương pháp này như `initialization` , `regularization` , hoặc mở rộng thành `Delaunay Terhedarization`
    
    <aside>
    💡
    
    Most reconstruction methods take it as an initialization [41] or regularization [42], or extend Delaunay tetrahedralization to remove the triangles of undesirable shape in reconstruction [6].
    
    </aside>
    

**Các phương pháp cụ thể sử dụng Triangulation-Based Prior**

1. **Greedy Delaunay (GD)**
    - Sử dụng greedy algorithm để chọn tam giác khả thi dựa trên các ràng buộc tô-pô *[**Topological Constraints** ](https://www.notion.so/Topological-Constraints-1462a3f8f2a180749d6beda19d4cccd5?pvs=21)*
    
    [What is a Greedy Algorithm? Examples of Greedy Algorithms](https://www.freecodecamp.org/news/greedy-algorithms/)
    
    - Các bước chính:
        1. Tạo tập hợp các tam giác ban đầu từ Delaunay triangulation.
        2. Chọn dần các tam giác thoả mãn điều kiện hình học và tô-pô.
2. **Ball-Pivoting Algorithm (BPA)**
    
    ![image.png](Surface%20Reconstruction%20From%20Point%20Clouds%20A%20Survey%20%201442a3f8f2a180aca990fd4b2793fb3b/image%204.png)
    
    [The Ball Pivoting Algorithm -  ppt video online download](https://slideplayer.com/slide/4773919/)
    
    - Phương pháp tạo tam giác bằng cách sử dụng **quả cầu lăn (rolling ball)**:
        - Một quả cầu có bán kính $r$ xác định lăn qua các điểm trong $P$.
        - Ba điểm $\{p_1, p_2, p_3\}$ được chọn tạo thành tam giác nếu quả cầu tiếp xúc đồng thời cả ba điểm mà không chứa điểm nào khác bên trong.
    - **Ưu điểm**: BPA gần giống với Delaunay nhưng linh hoạt hơn trong không gian 3D.

---

## Surface Smoothness Priors

![image.png](Surface%20Reconstruction%20From%20Point%20Clouds%20A%20Survey%20%201442a3f8f2a180aca990fd4b2793fb3b/image%205.png)

![image.png](Surface%20Reconstruction%20From%20Point%20Clouds%20A%20Survey%20%201442a3f8f2a180aca990fd4b2793fb3b/image%206.png)

Trong bài toán surface reconstruction, bề mặt $S$ cần được [`smooth`](https://www.notion.so/Surface-Reconstruction-From-Point-Clouds-A-Survey-and-a-Benchmark-1442a3f8f2a180aca990fd4b2793fb3b?pvs=21) hoặc ít nhất là `continuosly differentiable` một mức độ nào đó. Điều này giúp:

- **Giảm nhiễu (noise)**: Loại bỏ các dao động nhỏ hoặc bất thường trong dữ liệu.
- **Tạo hình dạng tự nhiên hơn**: Đảm bảo bề mặt không bị gồ ghề hoặc không thực tế.

Có ba strategies để áp đặt ràng buộc [smooth](https://www.notion.so/Surface-Reconstruction-From-Point-Clouds-A-Survey-and-a-Benchmark-1442a3f8f2a180aca990fd4b2793fb3b?pvs=21):

1. [**Surface Smoothness via Local Weighted Combination**](https://www.notion.so/Surface-Reconstruction-From-Point-Clouds-A-Survey-and-a-Benchmark-1442a3f8f2a180aca990fd4b2793fb3b?pvs=21)
2. [**Surface smoothness by combination of strategies**](https://www.notion.so/Surface-Reconstruction-From-Point-Clouds-A-Survey-and-a-Benchmark-1442a3f8f2a180aca990fd4b2793fb3b?pvs=21)
3. [**Surface smoothness by constraining function complexities**](https://www.notion.so/Surface-Reconstruction-From-Point-Clouds-A-Survey-and-a-Benchmark-1442a3f8f2a180aca990fd4b2793fb3b?pvs=21)

### **Local Weighted Combination**

Chiến lược này dựa trên ý tưởng rằng mỗi điểm $p$ trong point cloud sẽ được điều chỉnh dựa trên các điểm `local neighborhood` của $p$.

$$
\min_{f} \mathcal{L}_{\text{exp}}(f; P) = \frac{1}{n_P} \sum_{p \in P} \min_{x \in \Omega^2} \|f(x) - \hat{p}(p)\|^2
$$

**Giải thích các thành phần:**

- $n_P$: Số điểm trong tập $P$ (độ lớn của dữ liệu).
- $f(x)$: Hàm ánh xạ từ miền tham số $\Omega^2 \subset \mathbb{R}^2$đến không gian 3D, biểu diễn bề mặt $S$.
- $\hat{p}(p)$: Điểm được làm smooth của $p$, tính dựa trên các điểm lân cận $N(p)$.

**Cách tính $\hat{p}(p)$**:

$$
\hat{p}(p) = \sum_{p' \in N(p)} p' g(\|p - p'\|)
$$

**Giải thích:**

- $N(p)$: `local neighborhood` của điểm $p$, tức là tập hợp các điểm gần $p$ trong không gian 3D.
- $g(\|p - p'\|)$: Hàm trọng số, giảm giá trị khi khoảng cách $\|p - p'\|$ tăng. Một ví dụ phổ biến là [Radial Basis Function (RBF)](https://www.notion.so/Radial-Basis-Function-RBF-1462a3f8f2a180958de9e2e8325d6b64?pvs=21) :
    
    $$
    g(r) = \exp\left(-\frac{r^2}{2\sigma^2}\right)
    $$
    
    - $\sigma$: Điều chỉnh phạm vi ảnh hưởng của khoảng cách.

Hãy tưởng tượng bạn có một tập điểm 3D, trong đó một số điểm bị noise và không nằm đúng trên bề mặt $S$ thực sự. Để làm bề mặt smooth, bạn sẽ:

1. Xác định các điểm neighbor của mỗi điểm $p$ (bằng cách chọn các điểm trong bán kính nhất định, ví dụ $r = 0.1$).
2. Tính trung bình có trọng số của các điểm neighbor dựa trên khoảng cách của chúng đến $p$.
3. Sử dụng giá trị này để thay thế $p$, làm `smooth` nó.

**Poisson Surface Reconstruction (PSR)** là một phương pháp làm smooth bề mặt tương tự, áp dụng trong trường hợp sử dụng implicit representation bằng regularize smooth thông qua gradient của trường vector pháp tuyến:

$$
\mathcal{L}_{\text{PSR}}(F; P) = \mathbb{E}_{q \in \mathbb{R}^3} \|\nabla_q F(q) - \hat{n}(q; P)\|^2
$$

- $F(q)$: Hàm implicit bề mặt $S$ thông qua `zero-level set` $S = \{q \in \mathbb{R}^3 | F(q) = 0\}$
- $\nabla_q F(q)$: Gradient của $F$, thể hiện hướng pháp tuyến `normal vector` của bề mặt $S$ tại điểm $q$.
- $\hat{n}(q; P)$: Trường pháp tuyến `normal` trung bình tại q, được tính từ các điểm neighbor $N(q)$:

$$
\hat{n}(q; P) = \sum_{p' \in N(q)} n(p'; P) g(\|q - p'\|)
$$

- $n(p^{'}; P)$: Pháp tuyến tại điểm $p'$, thường được tính thông qua PCA từ neighbours.

**Ý nghĩa**: PSR cố gắng khớp gradient của $F(q)$ với local normal $\hat{n}(q; P)$, giúp bề mặt $S$ smooth hơn.

Vấn đề PSR thường tạo ra bề mặt quá smooth và mất đi các chi tiết details. **Screened Poisson Surface Reconstruction** (**SPSR)** khắc phục nhược điểm này bằng cách kết hợp điều chỉnh cả các regularizer bậc một và bậc hai trong hàm loss advanced implicit

<aside>
💡

As a remedy, Screened Poisson Surface Reconstruction (SPSR) [39] improves over PSR by incorporating regularized versions of both the first- and second-order terms in (7).

</aside>

Gần đây, https://pengsongyou.github.io/sap ra đời `is more interpretable, lightweight, and accelerates inference time by one order of magnitude.` End-to-end optimization, và có thể integrate deep neural networks. 

<aside>
💡

Shape As Points [44] develops a differentiable Poisson solver in a spectral manner, which enables an end-to-end optimization and thus could be integrated into the learning of deep neural networks.

</aside>

### Constraining function complexities

Bằng cách giới hạn [**Hypothesis Space**](https://www.notion.so/Hypothesis-Space-1452a3f8f2a18068a3b8d0832f1956fb?pvs=21) $H_f$ hoặc $H_{F}$ như `B-spline functions`, `NURBS`, `Gaussians` để tránh hiện tượng `overfiiting` - làm bề mặt ít smooth hơn.

Ví dụ:

1. Sử dụng hàm cơ bản [Gaussian (RBF)](https://www.notion.so/Surface-Reconstruction-From-Point-Clouds-A-Survey-and-a-Benchmark-1442a3f8f2a180aca990fd4b2793fb3b?pvs=21) [53] :  Áp dụng các đa thức bậc thấp để xấp xỉ implicit hoặc explicit function.
2. **Fourier Transform [54]**: sử dụng các hệ số Fourier để biểu diễn bề mặt và tái tạo bằng cách sử dụng **Divergence Theorem** https://youtu.be/vZGvgru4TwE?si=kpYY1cXCvvJXb-pS.
3. **Wavelet Representation [55]**: sử dụng wavelets để cung cấp mô hình đa độ phân giải (multi-resolution) nhằm cải thiện chi tiết.
4. Instead of using a regular grid, [50] optimizes the likelihood function over a finer grid under some topological constraints by solving the parameters of the Gaussian basis function.
5. He et al. [56] propose to regularize the curvatures of the mesh when reconstructing the surface, which results in a reconstructed surface that is smoother.
6. Wang [51] introduces a **Gaussian convolution** to the energy term, which promotes smoothness on the learned implicit function during the iterative optimization process.
7. **PGR** [52] chooses the basis of the indicator function that is guided by a Gaussian formulation to recover the surface from un-oriented points.

### Combination of Strategies

Chiến lược kết hợp (combination of strategies) trong việc đảm bảo smooth của bề mặt dựa trên sự tích hợp các phương pháp đã được đề cập, cụ thể là:

1. Kết hợp trọng số cục bộ (Local Weighted Combination).
2. Giới hạn độ phức tạp hàm (Constrained Function Complexities).

[www.cs.jhu.edu](https://www.cs.jhu.edu/~misha/Fall05/Papers/alexa01.pdf)

Point Set Surfaces Paper

Các phương pháp như **Point Set Surfaces (PSS)** tận dụng sức mạnh của cả hai strategies trên, tạo ra bề mặt smooth từ tập hợp các điểm observed $P$. PSS xuất phát từ lý thuyết **Moving Least Squares (MLS)**, là một kỹ thuật quan trọng trong việc xây dựng bề mặt từ đám mây điểm. MLS có thể được sử dụng để định nghĩa bề mặt theo:

1. **Hàm tường minh (explicit function)**: Xây dựng bề mặt qua các phép chiếu (project) trực tiếp lên không gian tham số.
2. **Hàm ngầm (implicit function)**: Đại diện bề mặt thông qua các giá trị mặt mức (level set).

PSS sử dụng **kết hợp trọng số cục bộ của các đa thức bậc thấp (spatially-varying low-degree polynomials)** để:

- Xấp xỉ các điểm quan sát $P$ trong một local neighbor.
- Xây dựng bề mặt bằng cách tối ưu hóa các đa thức này cục bộ tại mỗi điểm.

→ Các phương pháp như **SPSS, Algebraic PSS**, và **RIMLS** cải thiện hiệu suất, mở rộng khả năng tái tạo bề mặt trong các tình huống phức tạp hơn.

→ Cách tiếp cận thông qua **Moving Least Squares (MLS)** vẫn giữ vai trò nền tảng, kết hợp với các thuật toán cải tiến để đạt được độ chính xác cao hơn.

[Moving least squares](https://en.wikipedia.org/wiki/Moving_least_squares)

## Template-Based Priors

![image.png](Surface%20Reconstruction%20From%20Point%20Clouds%20A%20Survey%20%201442a3f8f2a180aca990fd4b2793fb3b/image%207.png)

**Template-based priors** giả định rằng bề mặt cần tái tạo có thể được biểu diễn bằng cách **kết hợp một nhóm các templates**. Các templates này có thể là:

- **Geometric primitives (hình học cơ bản)**: Hình cầu, hình hộp, hình trụ, hình nón, v.v.
- **Complex templates (mẫu phức tạp)**: Được lấy từ một tập dữ liệu phụ trợ (auxiliary dataset).

Quá trình surface reconstruction được `boil downs` thành bài toán:

1. Tìm các mẫu phù hợp từ tập templates đã cho.
2. Estimating/ fitting correct templates sao cho khớp với observed $P$.

Bài toán được định nghĩa dưới dạng:

$$
\min_{\{w\}, \{\theta\}} D\left( \sum_{i=1}^{|M|} w_i M_i(\theta_i), P \right)
$$

Trong đó:

- $\{M\} = \{M_1, M_2, \dots, M_{|M|}\}$: Tập hợp các templates đã được định nghĩa trước.
- $w_i$: Trọng số cho mỗi template $M_i$. Trọng số này có thể là biến số cần optimize hoặc đã được cố định.
- $M_i(\theta_i)$: Mỗi template $M_i$ được tham số hóa bởi $\theta_i$, các tham số này có thể được học `learnable parameters`.
- $D(\cdot, \cdot)$: Hàm khoảng cách đo sự khác biệt giữa bề mặt được tái tạo $\sum_{i} w_i M_i(\theta_i)$ và tập điểm quan sát $P$. Ví dụ: $D$ có thể là tổng bình phương khoảng cách Euclidean

**Ý nghĩa**: Khi minimize hàm $D$, ta tìm được tập trọng số $\{w\}$ và tham số $\{\theta\}$ để bề mặt tạo thành từ templates gần khớp nhất với tập điểm observed $P$.

**Các sub-methods cụ thể bao gồm:**

### **1. Geometric Primitives**

Templates đơn giản bao gồm các hình học cơ bản như: **Cuboids (hình hộp)**, **spheres (hình cầu)**, **cylinders (hình trụ)**, **cones (hình nón)**, v.v. Các hình học này có thể được biểu diễn chính xác bằng một số ít tham số (ví dụ: bán kính của hình cầu, chiều cao và bán kính của hình trụ).

**Giải thuật: Random Sample Consensus (RANSAC)**

[Dealing with Outliers: RANSAC | Image Stitching](https://youtu.be/EkYXjmiolBg?si=wNcAeIT3NDyILxEJ)

RANSAC là phương pháp phổ biến để giải bài toán $\min D(\cdot, P)$ với các templates hình học cơ bản:

1. **Ý tưởng**:
    - RANSAC lặp lại quá trình chọn ngẫu nhiên các điểm $p$ từ tập $P$ để khớp với một mẫu $M_i$.
    - Tính toán điểm số (score) dựa trên mức độ “khớp” của mẫu $M_i$ với dữ liệu quan sát.
    - Lựa chọn mẫu có điểm số cao nhất.
2. **Ví dụ trong nghiên cứu**:
    - **Schnabel et al. [65]**: Đề xuất thuật toán RANSAC hiệu quả với chiến lược chọn mẫu thông minh và đánh giá điểm số nhanh chóng.
    
    <aside>
    💡
    
    Schnabel et al. [65] introduce an efficient RANSAC-based algorithm with a novel sampling strategy and an efficient score evaluation scheme, where the primitive with maximum score is extracted iteratively.
    
    </aside>
    
    - **Nan và Wonka [66]**: Chỉ tập trung trích xuất các mẫu phẳng (planar primitives) từ RANSAC và tối ưu hóa tổng hợp bề mặt bằng cách sử dụng trọng số năng lượng (weighted sum of energy terms).
    
    <aside>
    💡
    
    Nan and Wonka [66] propose to only extract planar primitives based on RANSAC, obtaining the final surface by minimizing a weighted sum of several energy terms to select the optimal set of faces.
    
    </aside>
    
3. **Ứng dụng khác**: **Zhu et al. [67]**: Sử dụng các patch (mảnh nhỏ) từ point cloud và áp dụng các phương trình đạo hàm riêng (partial differential equations) để tái tạo bề mặt từ các templates khác nhau.
    
    <aside>
    💡
    
    Zhu et al. [67] reconstruct the surface by fitting the segmented point clouds with different primitive patches defined by the solution of the partial differentiable equation.
    
    </aside>
    

**Ưu điểm**: Phù hợp với các bề mặt có cấu trúc đơn giản hoặc các vật thể được mô hình hóa bằng các hình học cơ bản.

### 2. Retrieval-Based Templates

Phương pháp này sử dụng một tập template phụ trợ `auxiliary set` $\{M\}$, thường là các mô hình phức tạp được lưu trữ trước (auxiliary dataset), để:

1. Tìm kiếm `retrieve` các template gần đúng nhất.
2. Biến dạng `deform` các template này để khớp với tập điểm $P$.

**Cách thức hoạt động**:

- **Scene Reconstruction**:
    - Point cloud $P$ được segmented thành các `semantic classes`.
    - Mỗi class được khớp với một template đã lưu `retrieved template`, sau đó điều chỉnh bằng các phép `rigid` hoặc `non-rigid deformation`.

[www.uoanbar.edu.iq](https://www.uoanbar.edu.iq/eStoreImages/Bank/7116.pdf)

rigid and non-rigid deformation

- **Object Reconstruction**:
    - **Pauly et al. [72]**: Biến dạng và kết hợp nhiều mẫu từ tập phụ trợ để khớp với observed points.
    - **Shen et al. [73]**: Tái tạo bề mặt bằng cách trích xuất và lắp ráp các phần riêng lẻ của vật thể.

**Ví dụ trong nghiên cứu**:

- **Semantic Segmentation**: Trong tái tạo cảnh (scene reconstruction), phương pháp này có thể gán từng phần của $P$ vào các mẫu tương ứng với các vật thể cụ thể, chẳng hạn như bàn, ghế, hoặc cửa.
- **Lắp ráp mẫu (Part Assembly)**: Phù hợp với các vật thể phức tạp không thể biểu diễn bằng một mẫu duy nhất. Các phần nhỏ được gắn kết lại để tái tạo toàn bộ bề mặt.

**Ưu điểm**:

- Thích hợp cho các vật thể hoặc cảnh phức tạp có cấu trúc ngữ nghĩa.
- Cho phép tái tạo các bề mặt không thường xuyên hoặc không có mẫu hình học rõ ràng.

## Modeling Priors

**Modeling priors** là các ràng buộc hình học `geometry constraints` được cung cấp bởi chính thiết kế của các mô hình. Thay vì chỉ thêm các ràng buộc `regulizer` vào hàm loss hoặc hypothesis space, các thuộc tính của mô hình (ví dụ: mạng neural) được tận dụng để tự động áp đặt các constraints. Điều này giúp tái tạo bề mặt $S$ phù hợp với hình học của dữ liệu.

Một ví dụ tiêu biểu là **Deep Geometric Prior (DGP)**, được phát triển từ ý tưởng **deep image prior**. Cơ chế này cho thấy các mạng deep neural có khả năng tự động áp đặt các ràng buộc hình học ngay cả khi không được huấn luyện. **Deep Geometric Prior**

[Deep Image Prior](https://dmitryulyanov.github.io/deep_image_prior)

- Xác minh rằng các mạng neural có thể đóng vai trò như một prior hiệu quả cho mô hình hóa hình học mà không cần qua quá trình huấn luyện.
- Thay vì học từ dữ liệu, mạng được sử dụng như một công cụ để khớp dữ liệu quan sát với các cấu trúc hình học tự nhiên.

<aside>
💡

Motivated from deep image prior [74], Deep Geometric Prior (DGP) [75] verifies the efficacy of deep networks as a prior for geometric surface modeling, even when the networks are not trained.

</aside>

**Các phương pháp phát triển từ DGP:**

**a. Point2Mesh [76] và SAIL-S3 [77]**: Cả hai phương pháp mở rộng DGP từ cấp độ toàn cục (global modeling) sang cấp độ cục bộ (local modeling):

- **Point2Mesh**: Xây dựng các hàm ngầm cục bộ `global modeling adopted in [75] as local ones` dựa trên **MeshCNN**, trong đó các trọng số được chia sẻ giữa các phần của mạng.
- **SAIL-S3**: Sử dụng **Multi-Layer Perceptron (MLP)** với trọng số được chia sẻ để xây dựng các hàm ngầm cục bộ.

b. Deep Manifold Prior

- Phân tích toán học các đặc tính của MLP và mạng convolutional trong việc mô hình hóa bề mặt.
- Chỉ ra rằng bản chất của cấu trúc mạng neural, chẳng hạn như các layer ReLU, có thể hoạt động như một bộ regularizer.

Ngoài ra, **MLP và vai trò của ReLU trong Modeling Priors. Atzmon et al. [80] c**hứng minh lý thuyết rằng MLP với **Rectified Linear Units (ReLUs)** có thể tạo ra các bề mặt **piecewise linear**. **Analytic Marching Algorithm** [81] Đề xuất thuật toán để tính toán một cách giải tích các lưới tam giác (triangular mesh) từ cấu trúc của MLP, cho thấy cách thiết kế mạng neural có thể hành xử như một mô hình tái tạo bề mặt.

**Các phương pháp sử dụng Modeling Priors khác**

**a.  Implicit Geometric Regularization (IGR) [37]**: Áp dụng ràng buộc bổ sung lên gradient của mạng neural, giúp bề mặt được tái tạo trở nên mượt mà hơn.

**b. Sign Agnostic Learning (SAL) [82] và biến thể [83], [84]**:

- Tái tạo bề mặt từ point clouds không có định hướng (un-oriented point clouds) thông qua việc khởi tạo đặc biệt của mạng neural.
- **SSP [85]**: Sử dụng "semi-signed supervision" (giám sát bán ký hiệu), được tạo ra bằng cách voxel hóa các điểm không định hướng, nhằm tránh các cực tiểu cục bộ không tốt.

**c. Neural Splines [87]**:

- Tái tạo bề mặt dựa trên **random feature kernels** (hạt nhân đặc trưng ngẫu nhiên) từ các mạng ReLU nông với chiều rộng vô hạn.
- Chứng minh rằng cách tiếp cận này thiên về việc tái tạo các bề mặt smooth

## **Learning-Based Priors**

**Learning-Based Priors**: Là phương pháp học các **prior hình học** từ một tập dữ liệu huấn luyện gồm các cặp `learned from an auxiliary set of training shapes as optimized model parameters`, trong đó:

$$
\{P_i, S_i^*\}_{i=1}^N
$$

- $P_i$: Observed Point Sets tức là các tập hợp điểm $P_i = \{p_1, p_2, \dots, p_n\}$ thu thập từ các cảm biến 3D.
- $S_i^*$: ground-truth surfaces tương ứng với point cloud $P_i$ là bề mặt mà chúng ta muốn tái tạo.

**Ý tưởng chính**: Học các tham số mô hình ( $\theta_f$ hoặc $\theta_F$) từ tập dữ liệu để tái tạo bề mặt $S$ sao cho các đặc tính hình học được phù hợp.

**1. Biểu diễn tường minh (Explicit Representation)**

Trong trường hợp biểu diễn tường minh, chúng ta sử dụng hàm ánh xạ $f(z, \tilde{\theta}_f)$, với $\tilde{\theta}_f$ là các tham số đã học từ tập huấn luyện.

$$
\min_z \mathcal{L}_{\text{exp}}\big(f(z; \tilde{\theta}_f); P\big) + \lambda \mathcal{R}_{\text{exp}}(z)
$$

Trong đó:

- $f(z; \tilde{\theta}_f)$: Hàm ánh xạ tường minh với các tham số đã học $\tilde{\theta}_f$, giúp biến đổi latent code $z$ thành một surface $S$
- $z$: Latent code, biểu diễn hình dạng của bề mặt.
- $z$: Ràng buộc để tránh overfitting, thường là chuẩn hóa $\|z\|$.

**Quá trình học tham số $\tilde{\theta}_f$:**

$$
\tilde{\theta}_f = \arg\min_{\theta_f} \frac{1}{N} \sum_{i=1}^N \mathcal{\tilde{L}}_{\text{exp}}(\theta_f; \{P_i, S_i^*\}) + \tilde{\lambda} \mathcal{\tilde{R}}_{\text{exp}}(\theta_f)
$$

- **Ý nghĩa**:
    - Bước này học tham số mô hình $\theta_f$ bằng cách sử dụng toàn bộ tập dữ liệu huấn luyện sao cho bề mặt tái tạo gần nhất với bề mặt thực tế $S_i^*$ và tránh hiện tương overfiiting
    - $\mathcal{\tilde{L}}_{\text{exp}}$: Hàm mất mát trên tập huấn luyện (có thể khác với $\mathcal{L}_{\text{exp}}$).
    - $\tilde{\lambda}$: Điều chỉnh mức độ ưu tiên giữa overfitting và ràng buộc hình học.

**2. Biểu diễn ngầm (Implicit Representation)**

Công thức tối ưu hóa với implicit representation:

$$
\min_z \mathcal{L}_{\text{imp}}\big(F(z; \tilde{\theta}_F); P\big) + \lambda \mathcal{R}_{\text{imp}}(z)
$$

- $F(z; \tilde{\theta}_F)$: Hàm implicit, định nghĩa bề mặt qua zero-set level $F(q) = 0$ với các tham số đã học $\tilde{\theta}_F$

Các bước học $\tilde{\theta}_F$ tương tự như biểu diễn tường minh:

$$
\tilde{\theta}_F = \arg\min_{\theta_F} \frac{1}{N} \sum_{i=1}^N \mathcal{\tilde{L}}_{\text{imp}}(\theta_F; \{P_i, S_i^*\}) + \tilde{\lambda} \mathcal{\tilde{R}}_{\text{imp}}(\theta_F)
$$

1. **Auto-Encoder trong học tham số**

Một cấu trúc auto-encoder có thể được sử dụng để học tham số mô hình. Cấu trúc này bao gồm hai phần chính encoder và decoder:

- $f_{\text{encoder}} \circ f_{\text{decoder}}$: Kết hợp encoder và decoder.
- Sau khi học được $\tilde{\theta}_f = \{\tilde{\theta}_{f_{\text{encoder}}}, \tilde{\theta}_{f_{\text{decoder}}}\}$, latent code $z$ có thể được lấy trực tiếp từ:

$$
z = f_{\text{encoder}}(P; \tilde{\theta}_{f_{\text{encoder}}})
$$

Quy trình này cũng áp dụng tương tự cho hàm implicit $F$
:

$$
z = F_{\text{encoder}}(P; \tilde{\theta}_{F_{\text{encoder}}})
$$

**Ưu điểm của auto-encoder**:

- Tách biệt giữa việc học tham số và tối ưu hóa latent code, giúp quá trình học trở nên hiệu quả hơn.
- Giảm thiểu chi phí tính toán và tối ưu hóa.

**Hai cấp độ học Prior bao gồm:**

### **1. Semantic Priors**

**Semantic Priors** là việc học các đặc trưng ngữ nghĩa (semantic features) chung cho một danh mục đối tượng (category). Cụ thể, nó học các mô hình hình học cho các loại đối tượng cụ thể.

- **Ví dụ**:
    - **Deep Marching Cubes** [88]: Kết nối mạng neural với thuật toán Marching Cubes qua một lớp phân tích khả vi, giúp tái tạo bề mặt với prior ngữ nghĩa.
    - **OccNet** [9] và **IM-Net** [91]: Sử dụng auto-encoder để học global latent code, sau đó dự đoán khả năng tồn tại điểm trên bề mặt (**occupancy)** hoặc **signed distance**.
- **Ưu điểm**:
    - Dễ dàng khái quát hóa các hình dạng thuộc cùng một loại đối tượng.
    - Áp dụng được cho các bài toán với ít thông tin 3D, ví dụ: tái tạo từ hình ảnh RGB [5].

### **2. Local Shape Primitives**

Khi đối tượng không thể được phân loại vào danh mục cố định, phương pháp này học các **hình dạng cục bộ** (local shape primitives), có thể được áp dụng cho các đối tượng phức tạp hoặc không rõ ràng về ngữ nghĩa.

- Các đối tượng không thuộc danh mục cố định.
- Bề mặt phức tạp với nhiều chi tiết nhỏ.

**Ví dụ về phương pháp**:

- **IF-Net** [101]: Sử dụng 3D convolution để học latent code cho từng local grid.
- **DeepLS** [34]: Huấn luyện hàm implicit chia sẻ tham số giữa các local voxel.
- **PatchNets** [33]: Học các đặc trưng của từng patch bề mặt bằng cấu trúc implicit auto-decoder.

**Ưu điểm**:

- Linh hoạt hơn trong việc tái tạo các hình dạng không đồng nhất.
- Tăng khả năng mô hình hóa chi tiết nhỏ.

## **Hybrid Priors**

Trong phần này, tác giả thảo luận về các phương pháp sử dụng **ràng buộc hình học kết hợp** (*hybrid priors*), kết hợp nhiều chiến lược khác nhau để cải thiện độ tin cậy và tính hợp lý của bề mặt tái tạo (reconstructed surface). Cụ thể, mục tiêu là sử dụng các learning-based **priors** kết hợp với các **existing method** để đạt được kết quả tái tạo chính xác hơn và hợp lý hơn, đặc biệt là trong **deep learning era**. Dưới đây là phần phân tích chi tiết và giải thích các ví dụ mà tác giả đưa ra trong phần này.

**1. Các chiến lược kết hợp priors**

Các phương pháp trong lĩnh vực tái tạo bề mặt có thể được phân loại theo các loại **priors hình học** mà chúng sử dụng. Để cải thiện tính hợp lý của các bề mặt tái tạo, các phương pháp hiện tại thường **kết hợp nhiều priors**:

- **Kết hợp priors smoothness với priors triangulation hoặc template-based**.
- **Kết hợp các templated-based priors với các learning-based priors** để cải thiện chất lượng reconstruction.
- Các phương pháp kết hợp này trở nên phổ biến khi sử dụng **deep learning** cho tái tạo bề mặt từ các point cloud.

**2. Kết hợp priors học được với priors triangulation (Dựa trên lưới tam giác)**

Các phương pháp kết hợp priors học được và **Delaunay triangulation** là một ví dụ rõ ràng. Các deep-learning network được huấn luyện để rút ra các đặc điểm hình học từ point cloud và sử dụng chúng để tạo ra các triangular mesh cho bề mặt tái tạo.

**Điển hình: PointTriNet (Point Triangulation Network)**

- **PointTriNet** là một framework sử dụng **proposal network** để đưa ra các candidates of triangle và **classification network** để xác định xem các candidates này có phải là một phần của surface reconstruction hay không.
- Các bước hoạt động của PointTriNet:
    - **Proposal network**: Đề xuất các triangles từ các điểm trong Point Cloud Set.
    - **Classification network**: Dự đoán xem liệu một tam giác có nằm trong bề mặt tái tạo hay không.
    - **Iterative effect**: Hai mạng này làm việc tuần tự cho đến khi bề mặt cuối cùng được tạo ra dưới dạng lưới tam giác.

**→ Ý nghĩa**: Phương pháp này giúp tái tạo bề mặt từ đám mây điểm thông qua việc xây dựng các tam giác, sử dụng mạng học sâu để tối ưu hóa và chọn lựa các tam giác phù hợp.

**3. Delaunay Surface Elements (DSE)**

Một phương pháp khác là **Delaunay Surface Elements (DSE)**, kết hợp **Delaunay triangulation** với các **logarithmic maps** học được để tạo ra các mảnh tam giác nhỏ.

- Các **logarithmic maps** học được giúp điều chỉnh các đặc điểm hình học của bề mặt, đảm bảo bề mặt tái tạo hợp lý hơn.
- **Adaptive voting algorithm**: Một thuật toán bầu chọn thích nghi để chọn các tam giác ứng viên phù hợp từ các mảnh tam giác được tạo ra.

**4. DeepDT (Deep Delaunay Tetrahedrons)**

Phương pháp **DeepDT** sử dụng **deep learning** để học các đặc điểm hình học từ point cloud, sau đó kết hợp với thông graph structural information để xác định các **labels bên trong/ngoài** của **Delaunay tetrahedrons**. Sau đó, bề mặt được tái tạo bằng cách **lấy các mặt tam giác** giữa các tetrahedrons có **labels khác nhau** (inside/outside).

**→ Ý nghĩa**: Phương pháp này sử dụng các **Delaunay tetrahedrons** và các kỹ thuật học sâu để cải thiện việc xác định các mặt tam giác và bề mặt tái tạo.

**5. Tái tạo bề mặt với Dự đoán các vị trí đỉnh của tetrahedrons**

**Gao et al. (2019)** [127] sử dụng deep-learning network để dự đoán **offsets** (sự dịch chuyển) của các đỉnh của **tetrahedrons**, cùng với một mạng khác để dự đoán **occupancy** (tính đầy đủ) của từng tetrahedron. Bề mặt của vật thể được tạo ra từ **mặt của hai tetrahedrons** có **occupancy khác nhau**.

**→ Ý nghĩa**: Phương pháp này giúp dự đoán vị trí của các đỉnh và tính chất của các **tetrahedrons** để xây dựng bề mặt.

**6. Kết hợp priors học được với priors độ trơn (Smoothness Priors)**

Một cách kết hợp phổ biến là kết hợp **priors học được** với **priors độ trơn**:

- **Laplacian regularization**: Được sử dụng để đảm bảo bề mặt tái tạo có smooth, giảm thiểu các gồ ghề hoặc nhấp nhô.
- **Cosine smoothness**: Một dạng regularization khác giúp làm mịn bề mặt.

**Điển hình: IMLSNet**

- **IMLSNet** là một mạng học sâu sử dụng **implicit network** để tính toán khoảng cách có dấu (signed distances) tại các lưới trong không gian 3D theo mô hình **octree**.
- Độ trơn của bề mặt được điều chỉnh bằng cách định nghĩa các **signed distances** tương tự như trong **implicit moving least squares (MLS)**.

**7. Kết hợp priors học được với priors dựa trên template**

Các phương pháp kết hợp **priors học được** với **template-based priors**:

- **SPFN (Surface Prediction from Network)**: Đào tạo một deep-learning network để phù hợp với các **primitive geometry** (hình học nguyên thủy), ví dụ như các khối hình học cơ bản (hình cầu, hình hộp, v.v.).
- **CPFN (Combined Primitive Fitting Network)**: Học hai mạng SPFNs: một mạng cho các **primitive lớn** và một mạng cho các **primitive nhỏ**.
- **ParseNet**: Sử dụng một **neural decomposition module** để phân tách đám mây điểm thành nhiều phần con, mỗi phần được giả định là một **primitive** và được mô hình hóa bằng **B-splines**.

**8. Local Deep Implicit Function (LDIF)**

**LDIF** đại diện cho bề mặt dưới dạng **a set of shape elements**, mỗi phần tử được parameterized bằng các analytic shape variables và latent shape codes. **SIF encoder** và **PointNet encoder** được sử dụng để học các biến hình học và mã hóa ẩn.

**→ Ý nghĩa**: Phương pháp này cho phép tái tạo các bề mặt phức tạp và các hình học không gian lớn bằng cách sử dụng các mạng học sâu để học các đặc trưng hình học.

**9. RetrievalFuse**

**RetrievalFuse** sử dụng một bộ dữ liệu template các vật thể để **truy xuất** các mẫu từ một bộ auxiliary scene dataset và sử dụng một mạng deep-learning có **attention-based refinement** để tinh chỉnh bề mặt tái tạo từ các template.

**→ Ý nghĩa**: Phương pháp này sử dụng việc **truy xuất thông tin từ bộ dữ liệu hỗ trợ** `auxiliary dataset` để giúp tái tạo bề mặt cho các scences quy mô lớn.

**Details**

- [**Mesh 3d ? Type of Mesh ?**](https://www.notion.so/Mesh-3d-Type-of-Mesh-1462a3f8f2a180c39b65ef99f21c26a2?pvs=21)
- [**Piecewise Linear**](https://www.notion.so/Piecewise-Linear-1442a3f8f2a180a69fdce76deeec8af5?pvs=21)
- [**Locally Differentiable up to First Order**](https://www.notion.so/Locally-Differentiable-up-to-First-Order-1462a3f8f2a1809caedbdb095899ba6c?pvs=21)
- [**Locally Differentiable**](https://www.notion.so/Locally-Differentiable-1462a3f8f2a18068a642ee6ffb807c5f?pvs=21)
- [**Piecewise Linear**](https://www.notion.so/Piecewise-Linear-1462a3f8f2a180fab89dcbfae3d448da?pvs=21)
- [**Topological Constraints** ](https://www.notion.so/Topological-Constraints-1462a3f8f2a180749d6beda19d4cccd5?pvs=21)

---

# Benchmark

Tác giả bài báo đã đề xuất một **benchmarking pipeline** chi tiết để tạo ra dữ liệu tổng hợp (synthetic data) cho các bài toán tái tạo bề mặt (surface reconstruction) từ Point Cloud. Phần này tập trung vào dữ liệu các vật thể (object surfaces) và cảnh (scences) được tổng hợp với những đặc điểm sau:

1. Được tổ chức dựa trên level complex of surface.
2. Bao gồm cả dữ liệu quét lý tưởng (perfect scanning) và dữ liệu chứa các lỗi thường gặp trong thực tế.

Mục tiêu của benchmark là:

- Tạo ra bộ dữ liệu synthetic chuẩn hóa bao gồm các bề mặt vật thể có độ phức tạp khác nhau.
- Mô phỏng các lỗi thường gặp trong quá trình quét thực tế, bao gồm:
    - Nhiễu điểm (point-wise noise).
    - Phân bố không đồng đều (non-uniform distribution).
    - Điểm ngoại lai (point outliers).
    - Sai lệch vị trí quét (misalignment).
    - Mất điểm dữ liệu (missing points).

→ Bộ dữ liệu này cung cấp một môi trường để so sánh các thuật toán surface reconstruction một cách toàn diện.

---

## **The Synthetic Data of Object Surfaces**

![Screenshot 2024-11-23 at 16.11.12.png](Surface%20Reconstruction%20From%20Point%20Clouds%20A%20Survey%20%201442a3f8f2a180aca990fd4b2793fb3b/Screenshot_2024-11-23_at_16.11.12.png)

Section này trình bày quy trình xây dựng **dữ liệu tổng hợp (synthetic data)** nhằm hỗ trợ đánh giá các phương pháp surface reconstruction. Tác giả mô tả cách tạo ra dữ liệu từ các [CAD Model](https://www.notion.so/CAD-Model-1472a3f8f2a1809292f5c57c1f4a21a3?pvs=21) bằng cách giả lập nhiều tình huống scanning khác nhau để phản ánh các điều kiện thực tế. Bên cạnh đó, dữ liệu này được phân loại theo độ phức tạp bề mặt `complex surface` để phân tích hiệu năng của các phương pháp trong các trường hợp khác nhau. Dưới đây là giải thích và phân tích chi tiết từng nội dung quan trọng.

Tác giả đã xây dựng một **benchmark** toàn diện để đánh giá các phương pháp surface reconstruction. Bộ dữ liệu tổng hợp này bao gồm:

- Các CAD models với ba mức độ phức tạp bề mặt (`low`, `middle`, `high`).
- Các tình huống quét thực tế được mô phỏng chi tiết (5 dạng lỗi quét).

### **1. Thu thập và tiền xử lý dữ liệu CAD**

Tác giả sử dụng các **CAD models** từ các data repository *[**Dataset**](https://www.notion.so/Dataset-1452a3f8f2a18007b376d1ccbb94ed15?pvs=21)* hiện có:

- **Thingi10k [22]**: Một bộ dữ liệu gồm các mô hình in 3D được chia sẻ trực tuyến, với 3,000 objects được chọn ngẫu nhiên.
- **ABC [21]**: Chọn 3,000 thành phần công nghiệp từ tập **Chunk 0080**.
- **3DNet Cat200 [20]**: Gồm 3,400 hàng hóa thương mại.
- **Three D Scans [23]**: Gồm 106 tác phẩm điêu khắc nghệ thuật.

→ Tổng cộng có **9,506 CAD models** được thu thập.

1. **Loại bỏ các trường hợp không phù hợp**:
    - Một số **meshes** không đáp ứng điều kiện **watertight** (kín nước), **2D-manifold** (mặt 2D liên tục), hoặc quá phức tạp về mặt topology (**self-occlusion** và surface genus cao). ***[**Non-wateright, Self-occlusion, 2D -manifold,  Surface Genus**](https://www.notion.so/Non-wateright-Self-occlusion-2D-manifold-Surface-Genus-1472a3f8f2a18047bff1d503bbf5fa9d?pvs=21)***
    - Các phương pháp kiểm tra bao gồm:
        - Kiểm tra trạng thái **watertight** và **2D-manifold** [135].
        - Kiểm tra **self-occlusion** [136].
        - Tính toán **surface genus** [32].
2. **Normalization**:
    - Các **meshes** còn lại được đưa về trạng thái chuẩn hoá:
        - **Đặt tâm tại gốc tọa độ**.
        - **Co giãn isotropically** để nằm trong một hình cầu đơn vị. ***[**Istrophy**](https://www.notion.so/Istrophy-1472a3f8f2a1806d9de4f8d0d472316b?pvs=21)***

→ **Kết quả**: Sau khi lọc và xử lý, **1,620 object surface meshes** được giữ lại để đưa vào bộ dữ liệu **benchmark**.

### **2. Phân nhóm theo độ phức tạp bề mặt**

Tác giả phân tích rằng performance của các phương pháp surface reconstruction có thể phụ thuộc vào **độ phức tạp của bề mặt** `complexities of object surfaces`. Do đó, các đối tượng được chia thành ba nhóm dựa trên **algebraic complexity** (độ phức tạp đại số).

**Đo lường độ phức tạp [Algebraic Complexity, Topological Complexity](https://www.notion.so/Algebraic-Complexity-Topological-Complexity-1472a3f8f2a180d890bce97882e233d4?pvs=21)** 

- **Algebraic complexity**: Đo độ phức tạp đại số dựa trên **bậc cao nhất của đa thức (polynomial degree)** dùng để xấp xỉ các mảnh bề mặt cục bộ.
- **Topological complexity**: Được đo bằng **surface genus** (số lỗ trên bề mặt).

> Lưu ý: Các meshes đã được xử lý để đảm bảo:
> 
> - Watertight.
> - 2D-manifold.
> - Genus ≤ 5 (topology đơn giản).

**Phương pháp chia nhóm**

- **Khó khăn của việc đo lường**: Xấp xỉ bằng đa thức bậc cao thường không ổn định [60] và dễ gây lỗi dự đoán.
- **Giải pháp**: Tác giả sử dụng một chiến lược khác:
    - Giữ cố định bậc đa thức cao nhất.
    - Đo lường **sai số xấp xỉ trung bình** (average approximation errors) trên các mảnh bề mặt cục bộ.

<aside>
💡

However, given fixed budget of approximation errors, it is usually unstable to fit local surface patches with polynomials of high degrees [60], causing inaccurate prediction of polynomial degrees and thus that of algebraic complexity. Instead, we take the strategy of fixing the highest function degree and measuring the averaged approximation errors of local surface patches; technical details are given in Appendix A, available online.

</aside>

**Phân nhóm**

- Ba nhóm được chia theo tỷ lệ **6:3:1**:
    - **Low-complexity**: 972 instances.
    - **Middle-complexity**: 486 instances.
    - **High-complexity**: 162 instances.

### **3. Quét dữ liệu tổng hợp từ CAD models**

Tác giả sử dụng **Blender Sensor Simulation Toolbox (BlenSor) [137]** để mô phỏng các điều kiện quét dữ liệu, bao gồm cả `perfect scanning` và không hoàn hảo. Quy trình cụ thể được mô tả như sau:

**Perfect Scanning**

1. **Đặt camera TOF (Time-of-Flight)**:
    - Camera được đặt trên các **vòng cầu (viewing spheres)** có bán kính từ $r_{min}=2.5$ **đến $r_{max}=3.5$**.
    - Một điểm quan sát được xác định bởi **camera extrinsic matrix K = [R|t]**.
2. **Quét từ nhiều góc nhìn**:
    - Tác giả sử dụng **1,000 viewpoints** trên các vòng cầu để thu thập các point clouds bao phủ từng phần bề mặt.
    - Các point clouds riêng lẻ này được `register` và `fuse` để tạo ra một point cloud đầy đủ bao phủ toàn bộ bề mặt.
3. **Điều chỉnh density**: Áp dụng **Farthest Point Sampling (FPS)** để đảm bảo mật độ điểm đồng đều **[Farthest Point Sampling (FPS)](https://www.notion.so/Farthest-Point-Sampling-FPS-1472a3f8f2a1802ebe2de1bd22cfcdf5?pvs=21)** 
    - **80k points**: Low-complexity.
    - **120k points**: Middle-complexity.
    - **160k points**: High-complexity.
4. **Tính toán surface normals**: Sử dụng **PCA** trên lân cận **k=40 nearest neighbors** để tính toán các **normals**.

**Imperfect Scanning (5 loại lỗi quét):**

1. **Point-wise noise**: Mô phỏng **nhiễu điểm độc lập** bằng cách thêm nhiễu Gaussian với mức độ khác nhau $σ_{noise} = [0.001, 0.003, 0.006]$.
2. **Non-uniform distribution**: Sử dụng **Random Sampling (RS)** thay vì FPS để tạo mật độ điểm không đồng đều.
3. **Point outliers**: Thêm **outliers** bằng cách dịch chuyển một số điểm (tỷ lệ $outlier = [0.1\%, 0.3\%, 0.6\%]$) khỏi bề mặt với khoảng cách xác định $U[0.01, 0.1]$.
4. **Misalignment**: Mô phỏng **sai lệch camera extrinsic**:
    - Perturbation trong góc xoay [−0.5°, 0.5°] đến [−2°, 2°].
    - Perturbation trong dịch chuyển [−0.005, 0.005] đến [−0.02, 0.02].
5. **Missing points**: Mô phỏng mất điểm bằng cách giới hạn **viewing positions** trong các dải góc nhỏ `polar angle`:
    - Dải góc giảm `scanned surface areas` từ **99% (3 trajectories)** xuống **86% (1 trajectory)** tùy mức độ thiếu điểm.

---

## **The Synthetic Data of Scene Surfaces**

![Screenshot 2024-11-23 at 18.37.49.png](Surface%20Reconstruction%20From%20Point%20Clouds%20A%20Survey%20%201442a3f8f2a180aca990fd4b2793fb3b/Screenshot_2024-11-23_at_18.37.49.png)

Phần này mở rộng quy trình tạo dữ liệu tổng hợp từ cấp độ **object surfaces** (mô hình đối tượng) sang **scene surfaces** (mô hình cảnh), với mục tiêu mô phỏng việc quét bề mặt trong các môi trường thực tế quy mô lớn hơn. Tác giả xây dựng dữ liệu cấp độ cảnh với đầy đủ các dạng lỗi phổ biến trong quá trình quét thực tế, bao gồm **point-wise noise**, **point outliers**, **non-uniform distribution of points**, **misalignment**, và **missing points**. Phần này được mô tả chi tiết qua các bước từ thu thập dữ liệu đến quét tổng hợp, được phân tích bên dưới.

### **1. Thu thập dữ liệu CAD cho Scene Surfaces**

**Nguồn dữ liệu:** Tác giả thu thập **CAD surface models** từ ba bộ dữ liệu phổ biến, đảm bảo tính đa dạng về loại phòng, kích thước, nội thất, và vật dụng:

- **SceneNet [24]**: Chọn ngẫu nhiên **28 cảnh**.
- **3D-FRONT [25]**: Chọn **14 cảnh**.
- **Replica [29]**: Chọn **8 cảnh**.

**Đặc điểm dữ liệu (Tương tự như object)**

- Các **meshes** được thể hiện dưới dạng **triangular meshes** để đảm bảo tính nhất quán với dữ liệu cấp độ đối tượng.
- **Chuẩn hóa**:
    - **Căn giữa tại gốc tọa độ (origin)**: Đảm bảo mỗi cảnh được định vị ở trung tâm không gian.
    - **Giữ nguyên kích thước gốc**: Không co giãn kích thước như cấp độ object.

**Ý nghĩa:** Việc thu thập từ các bộ dữ liệu trên giúp tạo ra các cảnh thực tế đa dạng, phản ánh nhiều loại môi trường trong nhà.

### **2. Quy trình quét tổng hợp cho Scene Surfaces**

**Sử dụng công cụ quét**

- Công cụ quét tổng hợp: **Blender Sensor Simulation Toolbox (BlenSor)** [137].
- Cảm biến mô phỏng: **Kinect V2** [140] với khoảng cách làm việc từ **0.75m đến 2.1m**, phù hợp với các môi trường trong nhà.

**Cài đặt quét**

1. **Căn chỉnh và định vị scene**: Đặt mesh của **scene** vào **không gian 3D được giới hạn**:
    - Kích thước không gian: Bao trọn toàn bộ bề mặt của cảnh.
    - Tâm của cảnh được đặt tại gốc tọa độ.
2. **Phân chia không gian**:
    - Không gian 3D được chia thành các khối lập phương **kích thước 1m³**, chồng lấn nhau **0.5m³**.
    - Một số khối có thể rỗng (không chứa bề mặt), trong khi những khối khác chứa các mảnh bề mặt của cảnh.
3. **Đặt camera tại các vị trí khối rỗng**:
    - Chọn **tâm của các khối rỗng** làm vị trí đặt camera, đảm bảo camera nằm trong khoảng cách làm việc của Kinect V2.
    - Camera có thể quét các phần bề mặt từ **nhiều hướng khác nhau** trên một hình cầu quan sát (viewing sphere) xuất phát từ tâm của khối rỗng.
4. **Lấy mẫu ngẫu nhiên các hướng quét**:
    - Tại mỗi vị trí, chọn ngẫu nhiên **100 hướng quét** để thiết lập các thông số camera **extrinsics** :
        
        $$
        K_{ij} = [R_j | t_i]
        $$
        
        - $R_j$: Ma trận xoay (rotation matrix).
        - $t_i$: Vị trí camera.
5. **Tích hợp các point clouds**: Tạo ra **các point clouds cục bộ** từ mỗi hướng quét $K_{ij} \circ P_{K_{ij}}$, sau đó registeraion và hợp nhất chúng để tạo thành point cloud $P$ cuối cùng .

<aside>
💡

**Không áp dụng Farthest Point Sampling (FPS)**: Khác với level of object, ở cấp độ scences không áp dụng FPS, dẫn đến hiện tượng **non-uniform distribution** (phân bố không đồng đều) xảy ra tự nhiên.

</aside>

**Tính toán surface normals**: 

- Các **surface normals** được tính theo cách tương tự như trong phần **object scanning.**
- Dựa trên **PCA** từ 40 điểm lân cận gần nhất của mỗi điểm.

### **3. Mô phỏng các thách thức trong quét thực tế**

Cũng như complex of objects, dữ liệu tổng hợp complex of **scene** mô phỏng **5 main challenges** thường gặp trong thực tế:

**1) Non-uniform distribution of points**: **Tự nhiên xuất hiện** do không áp dụng FPS, dẫn đến mật độ điểm không đồng đều giữa các vùng bề mặt.

**2) Missing points**: Xuất hiện tự nhiên do **self-occlusion** (che khuất bản thân) và góc quan sát hạn chế của camera.

**3) Point-wise noise**: Mô phỏng nhiễu điểm độc lập: Nhiễu Gaussian với độ lệch chuẩn $\sigma_{\text{noise}} = 0.005m$

**4) Point outliers**: Thêm các điểm outliers bằng cách dịch chuyển một tỷ lệ $_{\text{outlier}} = 0.4\%$ của các điểm. Dịch chuyển ngẫu nhiên trong khoảng $[a_{\text{outlier}}, b_{\text{outlier}}] = [0.01m, 0.1m]$.

**5) Misalignment**: Mô phỏng sai lệch của **camera extrinsic**:

- Perturbation trong góc xoay: $[a_{\text{rotation}}, b_{\text{rotation}}] = [-1.5^\circ, 1.5^\circ]$
- Perturbation trong dịch chuyển: $[a_{\text{translation}}, b_{\text{translation}}] = [-0.015m, 0.015m]$

### **4. Kết quả của dữ liệu quét tổng hợp**

**Quy mô của point cloud cuối cùng:** Point cloud của mỗi cảnh chứa khoảng **1 triệu điểm**, bao gồm đầy đủ cả **5 challenges**.

**Ý nghĩa của dữ liệu cấp độ cảnh**

- Phản ánh các điều kiện thực tế của việc quét môi trường trong nhà với quy mô lớn.
- Dữ liệu này hỗ trợ đánh giá hiệu quả các phương pháp tái tạo bề mặt trong các điều kiện thực tế phức tạp hơn so với cấp độ đối tượng.

---

## **Section C: The Real-Scanned Data**

![Screenshot 2024-11-23 at 18.38.02.png](Surface%20Reconstruction%20From%20Point%20Clouds%20A%20Survey%20%201442a3f8f2a180aca990fd4b2793fb3b/Screenshot_2024-11-23_at_18.38.02.png)

Phần này tập trung vào dữ liệu quét thực tế (**real-scanned data**) được xây dựng để hỗ trợ đánh giá các phương pháp tái tạo bề mặt trong **benchmark**. Tác giả mô tả quy trình quét các đối tượng thực bằng hai loại camera quét 3D có độ chính xác khác nhau, nhằm tạo ra dữ liệu đầu vào và dữ liệu gần đúng với ground-truth. Dưới đây là phân tích chi tiết từng nội dung chính của phần này.

### **1. Tổng quan về dữ liệu quét thực tế**

**Mục tiêu**

- **Tạo dữ liệu quét thực tế** từ các đối tượng thật để bổ sung cho dữ liệu tổng hợp đã được mô tả trước đó.
- Bao gồm hai loại point clouds:
    - **Input point cloud**: Được quét bởi camera có độ chính xác thấp hơn, mô phỏng dữ liệu quét đầu vào thực tế.
    - **Ground-truth point cloud**: Được quét bởi camera có độ chính xác cao, sử dụng làm điểm tham chiếu (ground-truth) để đánh giá.

### **2. Thiết bị và thông số kỹ thuật**

**Thiết bị quét**

1. **SHINING 3D Einscan SE3**:
    - Độ chính xác: **100 micromet**.
    - Dùng để tạo **input point cloud**.
2. **SHINING 3D OKIO 5 M4**:
    - Độ chính xác: **5 micromet**.
    - Dùng để tạo **ground-truth point cloud**.

**Quy trình xử lý ground-truth**

- Ground-truth được xây dựng từ các point clouds quét bởi OKIO 5 M4, thay vì chuyển đổi chúng thành surface meshes bằng phần mềm tích hợp sẵn.
- **Lý do**: Đánh giá tái tạo bề mặt sử dụng các **metrics về khoảng cách giữa các tập điểm (point set distances)**, vì vậy sử dụng raw point clouds là hợp lý và trực tiếp hơn.

### **3. Thu thập dữ liệu**

**Đối tượng được quét**

Tác giả quét **20 đối tượng thực** với các đặc điểm đa dạng:

- **Độ phức tạp bề mặt**: Gồm các nhóm như hàng hóa (**commodities**), công cụ (**instruments**), và tác phẩm nghệ thuật (**artwares**).
- **Vật liệu bề mặt**: Kim loại (**metal**), nhựa (**plastic**), gốm (**ceramic**), và vải (**cloth**).

**Lý do lựa chọn đa dạng đối tượng và vật liệu vì Tính đại diện cao**: Đảm bảo rằng các điều kiện không hoàn hảo trong quét thực tế (imperfections) như phản xạ ánh sáng, che khuất, và phân bố không đồng đều được mô phỏng đầy đủ.

**Ví dụ minh họa**

- Các đối tượng thực tế được hiển thị trong **Hình 5** của bài viết.
- **Hình 6** minh họa các cặp dữ liệu quét (input vs. ground-truth) từ hai loại camera.

### **4. Quy trình quét thực tế**

**Quét từ nhiều góc nhìn**

- Chất lượng point clouds được cải thiện khi tăng số lần quét từ các góc nhìn khác nhau (**multi-view scanning**).
- **Hình 4** cho thấy mối quan hệ giữa số lượng góc quét và chỉ số **Chamfer Distances (CD)**: CD giảm khi số lượng góc quét tăng, nhưng đạt đến ngưỡng bão hòa sau một số lượng nhất định.

**Số lượng góc quét tối ưu**

- **Einscan SE**: Sử dụng khoảng **40 góc quét** để đạt chất lượng tối ưu.
- **OKIO 5 M**: Sử dụng khoảng **20 góc quét** để đạt chất lượng tối ưu.

**Công cụ xử lý:** Sau khi quét, các point clouds từ các góc quét khác nhau được căn chỉnh (**registration**) và hợp nhất bằng phần mềm **CloudCompare [141]**.

### **5. Đặc điểm dữ liệu quét thực tế**

**Đặc điểm nổi bật:** Các point clouds đầu vào có thể chứa các lỗi quét thực tế, như:

- Nhiễu (**noise**).
- Điểm ngoại lai (**outliers**).
- Sai lệch do căn chỉnh (**misalignment**).

**Thông số chi tiết**

- **Số lượng đối tượng**: 20.
- **Camera Einscan SE**:
    - Input point clouds.
    - Độ chính xác: 100 micromet.
    - 40 góc quét.
- **Camera OKIO 5 M**:
    - Ground-truth point clouds.
    - Độ chính xác: 5 micromet.
    - 20 góc quét.

**→ Thống kê dữ liệu:** Các thông số chi tiết của bộ dữ liệu quét thực tế được trình bày trong **Bảng IV**.

### **6. Mục tiêu của bộ dữ liệu quét thực tế**

**Đánh giá phương pháp tái tạo bề mặt**

- Cung cấp dữ liệu thực tế để kiểm tra hiệu năng của các phương pháp trong điều kiện thực tế.
- Đảm bảo rằng dữ liệu đại diện cho các lỗi quét phổ biến trong thực tế.

**So sánh với dữ liệu tổng hợp**

- Dữ liệu tổng hợp mô phỏng các lỗi quét trong môi trường giả lập.
- Dữ liệu thực tế kiểm tra khả năng áp dụng phương pháp trong điều kiện thực tế với các vật liệu và đối tượng phức tạp.

---

## **EXPERIMENTAL SET-UP FOR BENCHMARKING EXISTING SURFACE RECONSTRUCTION METHODS**

Section này trình bày cách thiết lập thực nghiệm nhằm **đánh giá các surface reconstruction hiện có** thông qua một bộ dữ liệu chuẩn (benchmark). Tác giả mô tả cách tổ chức dữ liệu, quy trình tiền xử lý, các tiêu chí đánh giá và các phương pháp được chọn để thử nghiệm. Mục tiêu là xác định ưu, nhược điểm của các phương pháp trong các điều kiện khác nhau.

### **A. Dữ liệu sử dụng trong thử nghiệm**

**1. Dữ liệu từ benchmark**

Từ bộ dữ liệu được xây dựng ở Section IV, tác giả chọn một tập con `subset` để thực hiện thử nghiệm:

- **22 surfaces tổng hợp của các object instances**:
    - Phân bố: **12 bề mặt có độ phức tạp thấp**, **6 trung bình**, **4 cao**.
    - Mỗi bề mặt được quét theo **6 cách** (bao gồm quét hoàn hảo `perfect scaning` và các lỗi quét khác nhau), tạo thành **308 cặp input-output** cho benchmarking.
- **50 bề mặt cảnh tổng hợp (synthetic scene surfaces)**.
- **20 bề mặt quét thực tế (real-scanned surfaces)**.

**2. Dữ liệu bổ trợ cho phương pháp học sâu**

Với các phương pháp dựa trên **learning-based priors**, tác giả cần một tập dữ liệu bổ trợ `aulixiary dataset` để huấn luyện các mô hình:

- **ShapeNet [18] và ABC [21]**: Các surfaces từ các nguồn này được dùng để huấn luyện.
- **Dữ liệu còn lại trong benchmark**: Sử dụng các dữ liệu không được chọn vào tập thử nghiệm.

→ Tác giả tiến hành huấn luyện lại (retrain) **OccNet [9]** và **DeepSDF [8]** trên tập dữ liệu bổ trợ này (chi tiết trong Appendix C).

### **B. Tiền xử lý dữ liệu**

Quy trình tiền xử lý được thực hiện trước khi áp dụng các phương pháp tái tạo bề mặt, nhằm cải thiện hiệu năng. Tác giả so sánh hiệu năng của các phương pháp **có** và **không có tiền xử lý**.

**1. Quy trình tiền xử lý dữ liệu tổng hợp**

- **Outlier removal (loại bỏ outliers)**:
    - Sử dụng phương pháp thống kê [142].
    - Quy tắc: Một điểm $p \in P$ được coi là outlier nếu **khoảng cách trung bình $\bar{d}_p$** của nó đến $k = 35$ điểm lân cận lớn hơn $5 \cdot \sigma_{\bar{d}}$ (độ lệch chuẩn của khoảng cách trung bình trên toàn bộ point cloud $P$).
- **De-noising (khử nhiễu)**:
    - Sử dụng **Jets smoothing [143]**:
        - Fit một bề mặt tham số cục bộ vào lân cận của điểm.
        - Dựng lại các điểm $\{p \in N\}$ bằng cách chiếu chúng lên bề mặt được fit.
        - Số lượng lân cận: $k = 18$.
- **Point re-sampling (lấy mẫu lại điểm)**:
    - Sử dụng **Farthest Point Sampling (FPS) [106]** để tạo phân bố điểm đồng đều hơn.
    - Giảm số lượng điểm còn **40%** của tập ban đầu.

**2. Tiền xử lý dữ liệu quét thực tế**

- Sử dụng công cụ tiền xử lý tích hợp của các máy quét.
- Sau đó thực hiện bước cuối: **FPS để lấy mẫu lại 200,000 điểm cho mỗi point cloud**.

### **C. Các tiêu chí đánh giá (Evaluation Metrics)**

Tác giả sử dụng 4 tiêu chí chính để định lượng và so sánh kết quả tái tạo:

**1. Chamfer Distance (CD) [145]**

- Đo **khoảng cách trung bình** từ các điểm trên bề mặt tái tạo đến bề mặt gốc và ngược lại.
- **Ý nghĩa**: Đo độ tương đồng hình dạng tổng thể giữa bề mặt tái tạo và bề mặt gốc.

**2. F-Score [146]**

- Đánh giá độ chính xác và độ bao phủ:
    - Precision: Tỷ lệ các điểm tái tạo nằm gần bề mặt gốc.
    - Recall: Tỷ lệ các điểm bề mặt gốc nằm gần bề mặt tái tạo.
- **Ý nghĩa**: Đo lường khả năng tái tạo bề mặt đúng và đầy đủ.

**3. Normal Consistency Score (NCS) [9]**

- Đo độ nhất quán giữa các pháp tuyến trên bề mặt tái tạo và bề mặt gốc.
- **Ý nghĩa**: Đánh giá chi tiết cục bộ, đặc biệt quan trọng với các phương pháp cần giữ được độ trơn của bề mặt.

**4. Neural Feature Similarity (NFS)**

- **Đề xuất mới**: Dựa trên sự tương đồng của hai bề mặt trong không gian đặc trưng sâu (**deep feature space**).
- **Ý nghĩa**: Đánh giá sự khác biệt ở mức độ ngữ nghĩa (semantic) gần gũi với nhận thức của con người.

**So sánh NFS với Light Field Distance (LFD) [148]**:

- Giống nhau: Cả hai đều có khả năng nắm bắt các chi tiết từ thô đến mịn của hình dạng.
- Khác nhau:
    - **NFS**: Áp dụng được cho tất cả các loại dữ liệu (synthetic và real-scanned).
    - **LFD**: Chỉ áp dụng được cho các bề mặt tổng hợp ở cấp độ đối tượng (synthetic object-level surfaces).

### **D. Các phương pháp được chọn và cách triển khai**

**1. Phương pháp được chọn**

Dựa trên các **geometric priors** đã phân loại trong Section III-E, tác giả chọn các phương pháp đại diện từ mỗi nhóm:

- **Triangulation-based prior**:
    - **GD [41]**.
    - **BPA [42]**.
- **Surface smoothness prior**:
    - **SPSR [39]**: Dựa trên chiến lược điều chuẩn bậc hai (cf. Eq. (10)).
    - **RIMLS [63]**: Kết hợp cả hai chiến lược điều chuẩn (cf. Eq. (9)).
- **Modeling prior**:
    - **SALD [83]**: Dành cho point clouds không định hướng (un-oriented).
    - **IGR [37]**: Dành cho point clouds có định hướng (oriented).
- **Learning-based prior**:
    - **OccNet [9]**, **DeepSDF [8]**: Học đặc trưng hình dạng toàn cục (global learning).
    - **LIG [10]**, **Points2Surf [89]**: Học đặc trưng cục bộ (local learning).
- **Hybrid priors**:
    - **DSE [123]**: Kết hợp triangulation-based và learning-based priors.
    - **IMLSNet [124]**: Kết hợp surface smoothness và learning-based priors.
    - **ParseNet [47]**: Kết hợp template-based và learning-based priors.

<aside>
💡

Tác giả không sử dụng phương pháp template-based prior bởi vì performance phần lớn phụ thuộc vào assumed templates và khó để reconstruct bề mặt phức tạp.

</aside>

**2. Cách triển khai**

- **Các phương pháp không dựa trên học sâu (classical methods)**:
    - **GD**: Dùng thư viện **CGAL [149]**.
    - **BPA, SPSR, RIMLS**: Sử dụng **MeshLab [150]** với điều chỉnh tham số phù hợp.
- **Các phương pháp học sâu (learning-based methods)**:
    - Sử dụng code được công bố chính thức của tác giả (nếu có).
    - Với các phương pháp không có mã nguồn, tác giả triển khai lại (re-implement) và tinh chỉnh hyper-parameters.

---

# **Main Results**

## **The Remaining Challenges**

Phần này tập trung vào việc phân tích các **thách thức chưa được giải quyết** trong bài toán tái tạo bề mặt (surface reconstruction) từ point clouds, dựa trên kết quả thực nghiệm từ các phương pháp so sánh. Mục tiêu của phần này là chỉ ra những hạn chế tồn tại trong các phương pháp hiện có, đặc biệt là khi đối mặt với các tình huống phức tạp và lỗi quét thực tế.

### **1. Tổng quan về các challenges còn tồn đọng**

**Các thách thức chính**:

1. **Misalignment** (Sai lệch định vị): Phát sinh khi các điểm từ các góc quét khác nhau không được căn chỉnh chính xác.
2. **Missing Points** (Điểm bị thiếu): Một phần của bề mặt không được quét hoặc không có dữ liệu.
3. **Point Outliers** (Điểm ngoại lai): Các điểm không nằm trên bề mặt thực tế.
4. **Point-wise Noise** (Nhiễu độc lập): Nhiễu gây dịch chuyển ngẫu nhiên của các điểm trong đám mây điểm.
5. **Non-uniform Point Distribution** (Phân bố điểm không đồng đều): Các khu vực quét có mật độ điểm khác nhau.

**Những quan sát nổi bật**:

- **Học sâu (Learning-based methods)**: Các phương pháp dựa trên deep learning có tiềm năng trong việc xử lý dữ liệu không hoàn hảo. Tuy nhiên, các thử nghiệm cho thấy chúng gặp khó khăn trong việc tổng quát hóa (generalization) cho các bề mặt phức tạp.
- **Phương pháp cổ điển (Classical methods)**: Các phương pháp như SPSR [39] thể hiện khả năng tổng quát tốt hơn trong một số trường hợp, đặc biệt khi dữ liệu chứa các lỗi.
- **Surface normals**: Việc sử dụng pháp tuyến bề mặt (surface normals) là yếu tố quan trọng để cải thiện chất lượng tái tạo, ngay cả khi các pháp tuyến được ước lượng không chính xác.

**Mâu thuẫn trong các chỉ số đánh giá**:

- Các chỉ số như **Chamfer Distance (CD)** và **F-score** đo lường hình dạng tổng thể nhưng có thể không phản ánh sự hài lòng về visualization.
- **Normal Consistency Score (NCS)** và **Neural Feature Similarity (NFS)** lại đánh giá tốt hơn mức độ chi tiết cục bộ và cảm nhận hình dạng.

### **2. Kết quả định lượng và thách thức từ các lỗi quét**

**Non-uniform Distribution of Points**:

- **Kết quả**: Hầu hết các phương pháp đều xử lý tốt thách thức này.
- **Ngoại lệ**: Các phương pháp dựa trên học sâu như DeepSDF [8], OccNet [9], và ParseNet [47] cho kết quả kém hơn.
- **Lý do**: Những phương pháp này học các đặc điểm hình học hoặc mẫu hình dạng (shape patterns), nhưng khi dữ liệu đầu vào không nằm trong miền đã học, hiệu suất suy giảm.

**Point-wise Noise**:

- **Quan sát**:
    - CD và F-score cho thấy các phương pháp vẫn có thể khôi phục hình dạng tổng thể.
    - Tuy nhiên, NCS và NFS giảm điểm, đặc biệt ở các phương pháp dựa trên triangulation như GD [41], BPA [42], và DSE [123].
- **Nguyên nhân**:
    - Các phương pháp triangulation phụ thuộc nhiều vào độ sạch của dữ liệu đầu vào.
    - Nhiễu làm giảm khả năng xác định các kết nối chính xác giữa các điểm.

**Misalignment**:

- **Kết quả**:
    - CD và F-score vẫn giữ được kết quả khả quan, nhưng NCS và NFS giảm mạnh.
    - Ví dụ minh họa (Fig. 8): Các phương pháp thường tạo ra các lớp bề mặt dày lên hoặc chồng lên nhau ở vùng sai lệch.
- **Vấn đề thực tế**:
    - Sai lệch định vị thường xảy ra khi sử dụng các máy quét cầm tay hoặc các thiết bị quét không chính xác.
- **Phương pháp tốt hơn**:
    - Các phương pháp sử dụng smoothness và modeling priors (e.g., SPSR [39]) có lợi thế trong việc xử lý sai lệch định vị.

**Missing Points**:

- **Kết quả**:
    - Hầu hết các phương pháp bỏ qua việc tái tạo ở các vùng bị thiếu điểm, dẫn đến bề mặt không đầy đủ (Fig. 9).
    - Phương pháp dựa trên **implicit modeling** (e.g., IGR [37], OccNet [9], Points2Surf [89]) tạo bề mặt kín (watertight) bằng cách điền vào các lỗ hổng. Tuy nhiên, điều này có thể dẫn đến bề mặt bị sai hình dạng, làm giảm chất lượng dưới tất cả các chỉ số đánh giá.
- **Kết luận**:
    - Đây là thách thức chưa được giải quyết bởi bất kỳ phương pháp nào.

**Point Outliers**:

- **Kết quả**:
    - Hiệu suất tái tạo phụ thuộc vào việc liệu phương pháp có cơ chế loại bỏ outliers hay không.
    - Ví dụ:
        - **BPA [42]**: Áp đặt điều kiện về kích thước của các tam giác, giúp xử lý tốt outliers.
        - **SPSR [39], IGR [37]**: Bỏ qua outliers một cách ngầm định (implicitly), giúp cải thiện hiệu suất.
- **Minh họa (Fig. 10)**: Kết quả cho thấy sự khác biệt rõ rệt giữa các phương pháp có và không có cơ chế xử lý outliers.

### **3. Tổng kết các phát hiện chính**

**Những thách thức chưa được giải quyết**:

1. **Missing Points**: Không có phương pháp nào giải quyết triệt để vấn đề thiếu dữ liệu.
2. **Misalignment và Point Outliers**: Hầu hết các phương pháp (ngoại trừ một số như SPSR [39]) đều cho kết quả không thỏa mãn.

**Mâu thuẫn giữa các chỉ số đánh giá**:

- Kết quả định lượng tốt (CD, F-score) không luôn đi đôi với sự hài lòng trực quan (NCS, NFS).
- Các phương pháp học sâu thường không tổng quát hóa tốt ngoài miền dữ liệu huấn luyện.

**Tầm quan trọng của Surface Normals**:

- Việc nhận diện bên trong/bên ngoài bề mặt trong không gian 3D là chìa khóa để cải thiện chất lượng tái tạo.

### **4. Tầm quan trọng của phân tích này**

Phần này giúp cộng đồng nghiên cứu nhận diện:

- Các điểm mạnh và hạn chế của từng loại phương pháp khi đối mặt với dữ liệu không hoàn hảo.
- Các vấn đề cần được tập trung giải quyết trong tương lai, như **missing points** và **misalignment**.
- Sự cần thiết của các chỉ số đánh giá tốt hơn để phản ánh chính xác chất lượng tái tạo bề mặt.

---

## **Optimization-Based, Learning-Free Methods versus Learning-Based, Data-Driven Ones**

Section này tập trung vào việc so sánh hai nhóm phương pháp trong bài toán tái tạo bề mặt (surface reconstruction):

1. **Optimization-Based, Learning-Free Methods**: Các phương pháp dựa trên tối ưu hóa và không sử dụng học sâu.
2. **Learning-Based, Data-Driven Methods**: Các phương pháp dựa trên học sâu và dữ liệu.

Mục tiêu là đánh giá khả năng tổng quát hóa (generalization), tính mạnh mẽ (robustness) trước dữ liệu không hoàn hảo và hiệu năng tái tạo các bề mặt phức tạp, bao gồm cả bề mặt có tính chất hình học (semantic shapes) và bề mặt không có cấu trúc rõ ràng (non-semantic shapes).

### **1. Phân loại hai nhóm phương pháp**

**A. Optimization-Based, Learning-Free Methods**

Nhóm này bao gồm các phương pháp cổ điển, không dựa vào learning-based:

- **Triangulation-Based Methods**: **GD [41]** và **BPA [42]**.
- **Surface Smoothness-Based Methods**:
    - **SPSR [39]** (Poisson Surface Reconstruction).
    - **RIMLS [63]**.
- **Modeling-Based Methods**:
    - **SALD [83]**: Phục hồi bề mặt từ point clouds không định hướng.
    - **IGR [37]**: Phục hồi từ point clouds có định hướng.

**→ Mục tiêu**: Kiểm tra tính tổng quát hóa và khả năng làm việc trên các bề mặt phức tạp mà không cần huấn luyện dữ liệu.

**B. Learning-Based, Data-Driven Methods**

Nhóm này sử dụng deep neural để tái tạo bề mặt từ dữ liệu đầu vào:

- **Semantic Global Learning Methods**:
    - **OccNet [9]** và **DeepSDF [8]**: Tái tạo dựa trên các đặc trưng hình học toàn cục (global semantic learning).
- **Local Learning Methods**:
    - **LIG [10]**, **Points2Surf [89]**, và **IMLSNet [124]**: Tái tạo bằng cách mô hình hóa và tổng hợp cục bộ.
- **Hybrid Priors**:
    - **DSE [123]**, **IMLSNet [124]**, và **ParseNet [47]**: Kết hợp học sâu với các priors hình học.

**→ Mục tiêu**: Đánh giá khả năng tận dụng dữ liệu học bổ trợ và xử lý dữ liệu không hoàn hảo.

### **2. Hiệu năng tổng quát hóa trên bề mặt phức tạp**

**A. Testing trên Synthetic Object Surfaces**

- **Dữ liệu thử nghiệm**:
    - Bao gồm 22 object instances (được mô tả trong Section V-A).
    - Gồm cả các đối tượng phổ biến có semantics rõ ràng (e.g., **chairs**) và các đối tượng không có cấu trúc hình học cụ thể (**non-semantic shapes**).
- **Kết quả định lượng (Table VIII)**:
    - **Optimization-Based Methods**:
        - Tổng quát hóa tốt hơn trên cả semantic và non-semantic shapes.
        - Hiệu năng ổn định trên tất cả các chỉ số (Chamfer Distance, F-Score, NCS, NFS).
    - **Learning-Based Methods**:
        - Hiệu năng tốt trên các semantic shapes (ví dụ: **chairs**) khi training set chứa các đối tượng thuộc cùng semantic categories.
        - Thất bại khi thử nghiệm trên non-semantic shapes hoặc các object không thuộc miền dữ liệu đã học.

**Minh họa (Fig. 12)**:

- **Optimization-Based Methods**: Tái tạo tốt hơn trên các bề mặt phức tạp hoặc không quen thuộc.
- **Learning-Based Methods**: Tái tạo semantic shapes chính xác, nhưng không tổng quát hóa được với các dạng dữ liệu mới.

### **3. Tính robustness trước dữ liệu không hoàn hảo**

**A. Nhận xét về Robustness**

- **Learning-Based Methods**:
    - Tỏ ra **`Robustness`** hơn khi dữ liệu đầu vào **ít thông tin hơn** (e.g., dữ liệu sparse). Điều này phù hợp với các nghiên cứu trước [11].
    - Tuy nhiên, khả năng tổng quát hóa vẫn bị hạn chế bởi miền dữ liệu huấn luyện.
- **Optimization-Based Methods**:
    - Ổn định hơn với dữ liệu không hoàn hảo, đặc biệt khi xử lý các lỗi quét như **point outliers** hoặc **misalignment**.

**B. Testing trên Real-Scanned Data**

- **Kết quả (Table VII, được tái tổ chức theo nhóm phương pháp)**:
    - Quan sát tương tự với synthetic data.
    - **Optimization-Based Methods** như SPSR [39] tiếp tục thể hiện độ tin cậy và hiệu năng ổn định.

### **4. Reconstruction trên Scene-Level Surfaces**

**A. Testing trên Synthetic Scene Surfaces**

- **Dữ liệu thử nghiệm**: Kết quả định lượng được báo cáo trong **Table VI** và minh họa trong **Fig. 11**.
- **Kết quả**:
    - **Local Learning Methods (e.g., LIG, Points2Surf, IMLSNet)**:
        - Xử lý tốt reconstruction trên scene-level surfaces bằng cách tổng hợp cục bộ các đặc trưng hình dạng.
    - **Global Semantic Learning Methods (e.g., OccNet, DeepSDF)**:
        - Thất bại trong việc tái tạo các bề mặt ở cấp độ cảnh, do thiếu khả năng tổng hợp cục bộ.

**B. Nguyên nhân thất bại của Semantic Global Learning Methods**

- Các phương pháp này chỉ hiệu quả khi dữ liệu đầu vào thuộc cùng miền (domain) với dữ liệu học.
- Không có khả năng xử lý tốt các hình dạng phức tạp không có semantics rõ ràng hoặc các bề mặt lớn hơn.

### **5. Kết luận chính**

![Screenshot 2024-11-23 at 18.52.29.png](Surface%20Reconstruction%20From%20Point%20Clouds%20A%20Survey%20%201442a3f8f2a180aca990fd4b2793fb3b/Screenshot_2024-11-23_at_18.52.29.png)

**Tóm tắt các phát hiện chính**:

1. **Optimization-Based Methods**:
    - Tỏ ra mạnh mẽ hơn trong tổng quát hóa trên cả object-level và scene-level surfaces.
    - SPSR [39] đặc biệt tốt về độ bền vững và khả năng xử lý các lỗi quét.
2. **Learning-Based Methods**:
    - **Global Semantic Learning (e.g., OccNet, DeepSDF)**:
        - Phù hợp với các object-level surfaces thuộc miền dữ liệu đã học.
        - Thất bại trên các hình dạng không quen thuộc và scene-level surfaces.
    - **Local Learning (e.g., LIG, Points2Surf)**:
        - Hiệu quả hơn trong tái tạo các bề mặt ở cấp độ cảnh nhờ khả năng tổng hợp cục bộ.
3. **Robustness trước dữ liệu không hoàn hảo**:
    - Các phương pháp học sâu tốt hơn khi dữ liệu đầu vào chứa ít thông tin.
    - Tuy nhiên, các phương pháp không dựa trên học sâu vẫn ổn định hơn khi đối mặt với các loại lỗi phức tạp.

### **6. Ý nghĩa đối với cộng đồng nghiên cứu**

- **Phương pháp học sâu (Learning-Based)**:
    - Cần cải thiện khả năng tổng quát hóa, đặc biệt khi làm việc với các bề mặt không thuộc miền dữ liệu học.
    - Local learning methods là một hướng đi triển vọng cho scene-level reconstruction.
- **Phương pháp cổ điển (Optimization-Based)**:
    - Tiếp tục là giải pháp đáng tin cậy cho các ứng dụng yêu cầu độ bền cao và khả năng xử lý dữ liệu không hoàn hảo.

---

## **The Importance of Orientations of Surface Normals**

Phần này tập trung vào vai trò quan trọng của **surface normals** trong quá trình tái tạo bề mặt từ point clouds. Tác giả thực hiện các thí nghiệm để kiểm tra cách việc tính toán hoặc ước lượng surface normals ảnh hưởng đến chất lượng tái tạo, đặc biệt khi dữ liệu đầu vào chứa các lỗi quét. Tóm tắt các phát hiện chính

1. **Surface normals** cải thiện chất lượng tái tạo bề mặt từ point clouds, đặc biệt khi dữ liệu không hoàn hảo.
2. **Hướng inward/outward của surface normals** quan trọng hơn độ chính xác tuyệt đối của chúng.
3. Hiệu năng giảm mạnh khi không có relative camera pose, do hướng pháp tuyến bị định hướng sai.
4. Các phương pháp sử dụng surface normals trong cả tính toán và học sâu cần chú ý đến yếu tố này để cải thiện hiệu suất.

Dưới đây là phân tích chi tiết và đầy đủ nội dung.

### **1. Tổng quan về việc sử dụng surface normals**

Tác giả liệt kê các phương pháp tận dụng **surface normals** trong quá trình tái tạo bề mặt, với các cách sử dụng khác nhau:

1. **Tính toán từ dữ liệu quan sát** (on-the-fly computation):
    - Các phương pháp như **BPA [42]**, **SPSR [39]**, **RIMLS [63]**, **IGR [37]**, **OccNet [9]**, **DeepSDF [8]**, **LIG [10]**, và **ParseNet [47]** đều tính toán surface normals từ point clouds quan sát được.
    - Surface normals được tính bằng cách sử dụng **Principal Component Analysis (PCA)** trên lân cận cục bộ của từng điểm.
2. **Ước lượng thông qua học sâu** (learning-based estimation):
    - Một số phương pháp như **Points2Surf [89]** và **IMLSNet [124]** tính toán surface normals từ dữ liệu huấn luyện và sau đó huấn luyện mô hình để ước lượng surface normals trên dữ liệu kiểm tra (testing data).

### **2. Vai trò của surface normals**

Tác giả muốn kiểm tra:

- Việc tính toán hoặc ước lượng **surface normals** có cải thiện chất lượng tái tạo hay không, đặc biệt khi dữ liệu đầu vào không hoàn hảo.
- Tầm quan trọng của **hướng inward/outward** của surface normals trong quá trình tái tạo.

### **3. Phương pháp thực nghiệm**

**Tính toán surface normals**:

- Giả sử vị trí tương đối (relative pose) của camera đối với đám mây điểm quan sát được.
- Surface normals được tính từ các điểm lân cận cục bộ bằng **PCA**:
    - **Local neighborhood**: Lựa chọn các điểm gần nhất (nearest neighbors).
    - **Oriented normal**: Hướng pháp tuyến được định hướng (inward hoặc outward) dựa trên tương quan với vị trí camera (cf. Section IV-A3).

**Trường hợp không có relative camera pose**:

- Khi không biết vị trí tương đối của camera, surface normals có thể bị định hướng sai trong các vùng cục bộ (local neighborhoods):
    - Điều này do **hướng chính (principal directions)** của PCA trên dữ liệu nhiễu không đáng tin cậy.

### **4. Kết quả thực nghiệm**

![Screenshot 2024-11-23 at 18.47.09.png](Surface%20Reconstruction%20From%20Point%20Clouds%20A%20Survey%20%201442a3f8f2a180aca990fd4b2793fb3b/Screenshot_2024-11-23_at_18.47.09.png)

**So sánh các phương pháp dựa trên surface normals**

Bảng **Table IX** tóm tắt kết quả định lượng trên dữ liệu tổng hợp của object surfaces:

- **Cột phân loại**:
    - **Không sử dụng surface normals** $(\times)$.
    - **Sử dụng surface normals** $(\checkmark)$.
    - **Chỉ sử dụng surface normals trong quá trình học** $(*)$
- **Kết quả**:
    - Các phương pháp sử dụng surface normals $(\checkmark)$ cho kết quả vượt trội hơn các phương pháp không sử dụng $(\times)$
    - Tuy nhiên, nếu hướng inward/outward của surface normals không chính xác (không có relative pose), hiệu suất giảm mạnh (kết quả bên phải của ký hiệu “/” trong bảng).

**Hiện tượng tương tự trên các dữ liệu khác**: Kết quả từ dữ liệu tổng hợp của **scene surfaces** và dữ liệu quét thực tế (**real-scanned data**) cũng cho thấy hiện tượng tương tự (chi tiết trong Appendix E):

- Khi hướng pháp tuyến không chính xác, hiệu suất tái tạo giảm mạnh.

### **5. Phân tích kết quả**

**Tầm quan trọng của surface normals**: Việc tính toán hoặc ước lượng surface normals là cần thiết để cải thiện chất lượng tái tạo bề mặt từ point clouds, ngay cả khi dữ liệu đầu vào không hoàn hảo.

**Hướng inward/outward** quan trọng hơn độ chính xác tuyệt đối: Tác giả nhấn mạnh rằng:

- **Hướng inward/outward chính xác** của surface normals quan trọng hơn so với độ chính xác tuyệt đối của các vector pháp tuyến.
- Lý do: Hướng đúng giúp phân biệt không gian bên trong và bên ngoài của bề mặt trong không gian 3D, hỗ trợ quá trình tái tạo.

### **6. Ý nghĩa của phát hiện**

**Đối với các phương pháp sử dụng surface normals**: Các phương pháp như **SPSR**, **RIMLS**, hoặc **IGR** có thể hưởng lợi lớn từ việc tính toán surface normals, miễn là hướng pháp tuyến được định hướng chính xác.

**Đối với phương pháp deep-learnning**: Các phương pháp học sâu như **Points2Surf** hoặc **IMLSNet** cần đảm bảo rằng mô hình có thể học được các đặc trưng phù hợp để định hướng surface normals trong quá trình huấn luyện.

---

# FAQ ?

### Observed data

Observed data are considered **a sample from an underlying large population** and is subject to random errors of measurement, specification errors, and other sources of noise, which in turn allows for an estimation of the impact of noise.

[](https://www.sciencedirect.com/topics/mathematics/observed-data)

---

### **Normal**

[3D basics - What are Normals?](https://youtu.be/L3t20_zYdOc?si=o_0Mvf_6NGJZ-c0s)

---

### **Robustness**

In more detail, robustness refers to the ability of a system, model, or entity to maintain stable and reliable performance across a broad spectrum of conditions, variations, or challenges, demonstrating resilience and adaptability in the face of uncertainties or unexpected changes.

---

### **Continuous Surface**

[Introductions-to-surface-continuities-and-its-types Blogs, Skill-Lync - Best job leading advanced online engineering courses provider with the most effective learning system in the world](https://skill-lync.com/blogs/introductions-to-surface-continuities-and-its-types)

[Discrete and continuous data—ArcMap | Documentation](https://desktop.arcgis.com/en/arcmap/latest/extensions/3d-analyst/discrete-and-continuous-data-in-3d-analyst.htm)

---

### Smooth Surface

Fair surfaces should follow the principle of *simplest shape*: the surface should be free of any unnecessary details or [oscillations](https://en.wikipedia.org/wiki/Oscillation).

[Surface fairing](https://en.wikipedia.org/wiki/Surface_fairing)

---

### **Point Cloud, Polygon meshes, Quantized Volumes ?**

- **Point Cloud**
    
    
    ![image.png](Surface%20Reconstruction%20From%20Point%20Clouds%20A%20Survey%20%201442a3f8f2a180aca990fd4b2793fb3b/image%208.png)
    
    ![image.png](Surface%20Reconstruction%20From%20Point%20Clouds%20A%20Survey%20%201442a3f8f2a180aca990fd4b2793fb3b/image%209.png)
    
    **Point Cloud** là tập hợp các điểm rời rạc  $P = \{p_1, p_2, \dots, p_n\}$ trong không gian 3D, trong đó mỗi điểm $p_i$ được biểu diễn bằng tọa độ $(x,y,z)$. Các điểm này được tạo ra bởi các thiết bị quét 3D như **LiDAR**, **Time-of-Flight**, hoặc **Structured Light Camera**.
    
    ### **Chức năng**
    
    - Các điểm trong Point Cloud mô tả vị trí của bề mặt vật thể một cách gần đúng, nhưng không bao gồm thông tin về các kết nối hoặc cấu trúc bề mặt giữa các điểm.
    - Đây là dạng dữ liệu đầu vào phổ biến nhất trong các ứng dụng tái dựng bề mặt.
    
    ### **Ưu điểm**
    
    - **Dễ thu thập**: Point Cloud là dạng dữ liệu gốc mà hầu hết các thiết bị quét 3D đều tạo ra trực tiếp.
    - **Linh hoạt**: Thích hợp cho việc lưu trữ và xử lý dữ liệu trong các ứng dụng như robot, bản đồ, và thực tế ảo.
    
    ### **Hạn chế**
    
    - **Không có thông tin kết nối**: Vì chỉ có tọa độ điểm, không có thông tin về cách các điểm tạo thành bề mặt.
    - **Đòi hỏi xử lý thêm**: Phải tái dựng bề mặt từ đám mây điểm để ứng dụng thực tế.
    
    ### **Ví dụ**
    
    Hãy tưởng tượng bạn quét một chiếc bàn bằng LiDAR. Kết quả là một tập hợp các điểm rải rác nằm trên mặt bàn, nhưng bạn không biết chúng liên kết với nhau như thế nào để tạo thành một bề mặt phẳng.
    
- **Polygon Mesh (Lưới đa giác)**
    
    ![image.png](Surface%20Reconstruction%20From%20Point%20Clouds%20A%20Survey%20%201442a3f8f2a180aca990fd4b2793fb3b/image%2010.png)
    
    ![image.png](Surface%20Reconstruction%20From%20Point%20Clouds%20A%20Survey%20%201442a3f8f2a180aca990fd4b2793fb3b/image%2011.png)
    
    **Polygon Mesh** là một cách biểu diễn bề mặt dưới dạng tập hợp các **đa giác (polygon)**, chủ yếu là các tam giác hoặc tứ giác, được kết nối với nhau. Một Polygon Mesh bao gồm hai phần chính:
    
    1. **Tập hợp đỉnh (Vertices)**: Các điểm 3D, tương tự như trong Point Cloud.
    2. **Tập hợp các đa giác (Polygons)**: Mỗi đa giác là một tập hợp các đỉnh được kết nối với nhau bằng các cạnh.
    
    ### **Chức năng**
    
    - Polygon Mesh cung cấp cấu trúc kết nối giữa các điểm, từ đó tạo ra bề mặt liên tục gần đúng.
    - Mỗi đa giác biểu diễn một phần nhỏ của bề mặt.
    
    ### **Ưu điểm**
    
    - **Cung cấp thông tin hình học đầy đủ hơn**: Ngoài tọa độ các điểm, Polygon Mesh xác định cách các điểm kết nối để tạo thành bề mặt.
    - **Hiệu quả trong kết xuất đồ họa**: Các phần mềm 3D và game thường sử dụng lưới đa giác để hiển thị vật thể.
    
    ### **Hạn chế**
    
    - **Kích thước lớn**: Polygon Mesh có thể yêu cầu nhiều bộ nhớ hơn để lưu trữ thông tin về các kết nối.
    - **Khó thu thập trực tiếp**: Không như Point Cloud, Polygon Mesh thường được tạo ra sau quá trình xử lý dữ liệu.
    
    ### **Ví dụ**
    
    Nếu bạn quét một chiếc bàn, một Polygon Mesh sẽ không chỉ lưu trữ các điểm trên bề mặt mà còn mô tả cách chúng kết nối với nhau để tạo ra các tam giác nhỏ, từ đó tạo nên mặt bàn.
    
- **Quantized Volumes (Voxel)**
    
    
    ![image.png](Surface%20Reconstruction%20From%20Point%20Clouds%20A%20Survey%20%201442a3f8f2a180aca990fd4b2793fb3b/image%2012.png)
    
    ![image.png](Surface%20Reconstruction%20From%20Point%20Clouds%20A%20Survey%20%201442a3f8f2a180aca990fd4b2793fb3b/image%2013.png)
    
    ![image.png](Surface%20Reconstruction%20From%20Point%20Clouds%20A%20Survey%20%201442a3f8f2a180aca990fd4b2793fb3b/image%2014.png)
    
    **Quantized Volume** là cách biểu diễn bề mặt dưới dạng một **khối lượng (volume)** trong không gian 3D được chia nhỏ thành các **voxel** (khối lập phương nhỏ). Mỗi voxel là một đơn vị cơ bản, chứa thông tin về việc nó nằm trong, ngoài, hay trên bề mặt.
    
    ### **Chức năng**
    
    - Cung cấp một cách biểu diễn toàn bộ vật thể, không chỉ bề mặt.
    - Mỗi voxel có thể lưu trữ thông tin thêm, chẳng hạn như mật độ hoặc màu sắc.
    
    ### **Ưu điểm**
    
    - **Toàn diện**: Quantized Volume không chỉ biểu diễn bề mặt mà còn cả cấu trúc bên trong (nếu có).
    - **Đơn giản hóa tính toán**: Các phép toán trên voxel thường dễ triển khai nhờ tính dạng lưới đều.
    
    ### **Hạn chế**
    
    - **Tốn bộ nhớ**: Việc chia nhỏ không gian 3D thành các voxel đòi hỏi lượng lớn bộ nhớ, đặc biệt khi độ phân giải cao.
    - **Độ chính xác giới hạn**: Kích thước của mỗi voxel ảnh hưởng đến độ chính xác của mô hình.
    
    ### **Ví dụ**
    
    Nếu bạn quét một chiếc bàn, Quantized Volume sẽ chia không gian xung quanh bàn thành các voxel. Mỗi voxel sẽ được đánh dấu là "bên trong," "bên ngoài," hoặc "trên bề mặt" bàn.
    

| **Thuộc tính** | **Point Cloud** | **Polygon Mesh** | **Quantized Volume** |
| --- | --- | --- | --- |
| **Dữ liệu đầu ra** | Các điểm rời rạc | Các đa giác (tam giác, tứ giác) | Các voxel 3D |
| **Kết nối** | Không có kết nối | Có kết nối | Không cần kết nối |
| **Độ chính xác** | Phụ thuộc vào mật độ điểm | Cao (nhờ kết nối rõ ràng) | Phụ thuộc vào kích thước voxel |
| **Tốn bộ nhớ** | Ít | Trung bình | Cao |
| **Ứng dụng phổ biến** | Đầu vào của tái dựng bề mặt | Đồ họa 3D, hoạt hình | Phân tích thể tích, y học |

![image.png](Surface%20Reconstruction%20From%20Point%20Clouds%20A%20Survey%20%201442a3f8f2a180aca990fd4b2793fb3b/image%2015.png)

![image.png](Surface%20Reconstruction%20From%20Point%20Clouds%20A%20Survey%20%201442a3f8f2a180aca990fd4b2793fb3b/image%2016.png)

![image.png](Surface%20Reconstruction%20From%20Point%20Clouds%20A%20Survey%20%201442a3f8f2a180aca990fd4b2793fb3b/image%2017.png)

---

### **Multi-view Stereo, Structured Light, Time-of-flight ?**

- **Multi-view Stereo,**
    
    [What is Multi-view Stereo (MVS)](https://www.activeloop.ai/resources/glossary/multi-view-stereo-mvs/)
    
    **Multi-view Stereo (MVS)** là một kỹ thuật trong thị giác máy tính, sử dụng nhiều hình ảnh chụp từ các góc nhìn khác nhau để tái dựng mô hình 3D của một vật thể hoặc cảnh.
    
    ![image.png](Surface%20Reconstruction%20From%20Point%20Clouds%20A%20Survey%20%201442a3f8f2a180aca990fd4b2793fb3b/image%2018.png)
    
    ### **Cách hoạt động của MVS:**
    
    1. **Chụp nhiều hình ảnh**: Một hoặc nhiều camera được đặt ở các góc khác nhau để chụp cùng một vật thể hoặc cảnh.
    2. **Tìm điểm tương ứng giữa các hình ảnh**:
        - Thuật toán so sánh các pixel trong hình ảnh để xác định các điểm chung xuất hiện trên nhiều bức ảnh.
        - Dựa trên nguyên lý **thị sai (parallax)**, các điểm tương ứng trên các hình ảnh sẽ có vị trí hơi khác nhau do góc chụp khác biệt.
    3. **Tính toán tọa độ 3D**: Dùng kỹ thuật hình học (như **triangulation**) để tính toán tọa độ (x,y,z) của mỗi điểm trên bề mặt dựa vào vị trí camera và các điểm tương ứng.
    4. **Tái dựng bề mặt**: Sau khi có tập hợp các điểm 3D, chúng được xử lý để tạo ra point cloud hoặc mô hình lưới tam giác (mesh).
    
    ### **Ưu và nhược điểm:**
    
    - **Ưu điểm**: Không cần phần cứng đặc biệt; chỉ cần camera và phần mềm xử lý.
    - **Nhược điểm**: Hiệu suất phụ thuộc vào độ chính xác của việc tìm điểm tương ứng; khó hoạt động tốt trên bề mặt trơn bóng hoặc có kết cấu đồng nhất.
    
    ### **Ví dụ**:
    
    - MVS thường được sử dụng trong tái dựng các mô hình 3D từ ảnh chụp kiến trúc hoặc di tích lịch sử.
- **Structured light**
    
    [What is Structured Light Imaging?  | RoboticsTomorrow](https://www.roboticstomorrow.com/article/2018/04/what-is-structured-light-imaging/11821)
    
    ![image.png](Surface%20Reconstruction%20From%20Point%20Clouds%20A%20Survey%20%201442a3f8f2a180aca990fd4b2793fb3b/image%2019.png)
    
    **Structured Light** là một kỹ thuật sử dụng ánh sáng được chiếu có cấu trúc (pattern light) để thu thập dữ liệu độ sâu.
    
    Structured Light phổ biến trong máy quét 3D, ví dụ như các máy quét sử dụng trong công nghiệp để đo đạc chi tiết hoặc kiểm tra sản phẩm.
    
    ### **Cách hoạt động của Structured Light:**
    
    1. **Chiếu ánh sáng có hoa văn lên vật thể**:
        - Một nguồn ánh sáng đặc biệt (ví dụ: máy chiếu) chiếu các hoa văn như đường sọc, lưới, hoặc chấm lên bề mặt vật thể.
    2. **Ghi lại hình ảnh bị biến dạng**:
        - Camera ghi lại hình ảnh của hoa văn sau khi bị biến dạng bởi bề mặt vật thể.
    3. **Phân tích biến dạng**:
        - Dựa vào cách hoa văn bị biến dạng, thuật toán tính toán khoảng cách từ cảm biến đến các điểm trên bề mặt (giống như cách mắt người phân biệt độ sâu từ bóng).
    4. **Tạo point cloud**: Sử dụng dữ liệu khoảng cách để tạo ra mô hình point cloud.
    
    ### **Ưu và nhược điểm:**
    
    - **Ưu điểm**:
        - Chính xác và nhanh chóng.
        - Hoạt động tốt trên các bề mặt có kết cấu đồng nhất.
    - **Nhược điểm**:
        - Đòi hỏi điều kiện ánh sáng ổn định, ít nhiễu từ ánh sáng môi trường.
        - Cần phần cứng đặc biệt (máy chiếu và camera đồng bộ).
- **Time-of-flight**
    
    ![image.png](Surface%20Reconstruction%20From%20Point%20Clouds%20A%20Survey%20%201442a3f8f2a180aca990fd4b2793fb3b/image%2020.png)
    
    **Time-of-Flight (ToF)** là một kỹ thuật đo khoảng cách bằng cách tính thời gian ánh sáng đi từ nguồn phát đến bề mặt và quay lại cảm biến.
    
    ToF thường được dùng trong cảm biến đo khoảng cách trên robot, drone, hoặc điện thoại thông minh (ví dụ: cảm biến ToF trên iPhone hoặc Samsung để hỗ trợ AR).
    
    [Time of Flight Sensor vs. LiDAR: What Are the Differences?](https://pmt-fl.com/time-of-flight-sensor-vs-lidar-what-are-the-differences/)
    
    ### **Cách hoạt động của ToF:**
    
    1. **Phát tia sáng (hoặc xung ánh sáng)**:
        - Nguồn sáng (thường là tia laser hoặc LED hồng ngoại) phát xung ánh sáng đến vật thể.
    2. **Ghi nhận ánh sáng phản xạ**:
        - Cảm biến ghi lại ánh sáng phản xạ từ bề mặt vật thể.
    3. **Tính toán khoảng cách**:
        - Sử dụng công thức $d = \frac{c \cdot t}{2}$ trong đó:
            - d: Khoảng cách từ cảm biến đến bề mặt.
            - c: Vận tốc ánh sáng (~300,000 km/s).
            - t: Thời gian ánh sáng đi và về.
    4. **Tạo bản đồ độ sâu hoặc point cloud**:
        - Dựa trên các khoảng cách đo được, tạo ra mô hình 3D.
    
    ### **Ưu và nhược điểm:**
    
    - **Ưu điểm**:
        - Chính xác, phù hợp cho các môi trường phức tạp hoặc ánh sáng yếu.
        - Hoạt động tốt trên các vật thể không có kết cấu.
    - **Nhược điểm**:
        - Độ chính xác có thể giảm nếu ánh sáng bị hấp thụ hoặc tán xạ bởi vật thể.
        - Chi phí cao do yêu cầu phần cứng chuyên dụng.

| **Kỹ thuật** | **Nguyên lý chính** | **Yêu cầu phần cứng** | **Ứng dụng phổ biến** |
| --- | --- | --- | --- |
| Multiview Stereo | Phân tích hình ảnh từ nhiều góc | Camera | Tái dựng từ ảnh chụp |
| Structured Light | Phân tích ánh sáng chiếu lên bề mặt | Máy chiếu + Camera | Máy quét 3D công nghiệp |
| Time-of-Flight | Tính thời gian đi/về của ánh sáng | Nguồn sáng + Cảm biến ToF | Cảm biến đo khoảng cách AR |

---

### **Ill-posed**

[Well Posed and Ill Posed problems & Tikhonov Regularization](https://www.statisticshowto.com/well-posed-ill/)

Trong toán học, một bài toán được coi là **"well-posed"** nếu thỏa mãn ba điều kiện do Jacques Hadamard đưa ra:

1. **Tồn tại nghiệm**: Luôn có ít nhất một lời giải.
2. **Duy nhất nghiệm**: Chỉ có một lời giải duy nhất.
3. **Ổn định nghiệm**: Nghiệm thay đổi không đáng kể nếu dữ liệu đầu vào có sai số nhỏ.

Ngược lại, một bài toán là **"ill-posed"** nếu không thỏa mãn bất kỳ điều kiện nào trong ba điều kiện trên.

### **Tại sao bài toán surface reconstruction là ill-posed?**

- Khi chúng ta chỉ có dữ liệu từ một tập hợp điểm rời rạc (**point cloud**), không có cách duy nhất để suy luận bề mặt liên tục ẩn đằng sau tập hợp điểm đó. Có thể tồn tại **vô số bề mặt** khác nhau phù hợp với dữ liệu này.
- Ngoài ra, các dữ liệu đầu vào thường chứa lỗi (như nhiễu hoặc thiếu điểm), dẫn đến việc nghiệm trở nên không ổn định.

**Ví dụ minh họa**:

- Hãy tưởng tượng bạn có 5 điểm nằm trên mặt đất. Bạn có thể nối chúng thành một đường thẳng, một đường cong, hay một chuỗi các hình zigzag — tất cả đều hợp lý, vì không có thông tin rõ ràng nào định nghĩa hình dạng thực sự của bề mặt.

---

### **Surface Reconstruction Challenge ?**

### **1. Nhiễu (Noisy point clouds)**

**Ý nghĩa**: Trong thực tế, các cảm biến 3D như LiDAR hoặc máy quét laser không đo đạc hoàn hảo. Dữ liệu đo có thể bị sai lệch do các yếu tố như:

- Nhiễu từ môi trường (ánh sáng, nhiệt độ).
- Lỗi cảm biến (độ chính xác hạn chế của thiết bị).
- Phản xạ từ bề mặt (ví dụ: ánh sáng chiếu lệch gây sai số đo).

**Hệ quả**: Các điểm trong point cloud không nằm chính xác trên bề mặt thực, dẫn đến việc tái dựng bề mặt bị sai lệch.

**Ví dụ trực quan**: Thay vì một mặt phẳng mịn, dữ liệu đo được có thể trông như một tập hợp các điểm "lung lay" xung quanh mặt phẳng thực.

**Giải pháp thường dùng**:

1. **Lọc nhiễu (denoising)**: Sử dụng các thuật toán để loại bỏ hoặc giảm thiểu nhiễu.
2. **Regularization**: Áp dụng các ràng buộc toán học để làm mịn bề mặt tái dựng.

---

### **2. Phân bố không đồng đều (Non-uniform distribution)**

**Ý nghĩa**: Điểm trong point cloud không phải lúc nào cũng được phân bố đều trên toàn bộ bề mặt. Một số vùng có thể có nhiều điểm hơn, trong khi các vùng khác lại có rất ít hoặc không có điểm.

**Nguyên nhân**:

- Góc quét: Một số vùng được quét từ nhiều góc độ hơn.
- Khoảng cách: Cảm biến quét các vùng gần với độ chi tiết cao hơn.
- Vật cản: Một số vùng bị che khuất dẫn đến thiếu điểm.

**Hệ quả**: Các vùng thưa điểm khó tái dựng chính xác, vì thiếu thông tin để định hình bề mặt.

**Ví dụ**: Hãy tưởng tượng quét một quả táo: mặt trước của quả táo có nhiều điểm hơn vì hướng trực tiếp với máy quét, còn mặt sau (bị khuất) có ít điểm hơn.

**Giải pháp thường dùng**:

1. **Upsampling**: Tăng số lượng điểm ở các vùng thưa bằng cách nội suy.
2. **Kết hợp nhiều góc quét (multi-view scanning)** để thu thập đủ điểm.

---

### **3. Ngoại lệ (Outliers)**

**Ý nghĩa**: Outliers là các điểm trong point cloud **không thuộc bề mặt thực**, mà là các giá trị đo lỗi.

**Nguyên nhân**:

- Lỗi cảm biến: Một số điểm bị sai lệch lớn.
- Nhiễu từ ánh sáng hoặc vật thể khác: Ví dụ, cảm biến quét trúng một vật thể khác gần bề mặt thực.

**Hệ quả**: Nếu không xử lý, các outliers sẽ làm sai lệch kết quả tái dựng bề mặt.

**Ví dụ**: Trong quá trình quét một chiếc ghế, một vài điểm từ nền nhà vô tình được ghi nhận là thuộc về bề mặt của ghế.

**Giải pháp thường dùng**:

1. **Outlier detection**: Phát hiện và loại bỏ các điểm bất thường.
2. **Statistical filtering**: Lọc điểm dựa trên khoảng cách trung bình đến các điểm lân cận.

---

### **4. Sai lệch căn chỉnh (Misalignment)**

**Ý nghĩa**: Khi point clouds được tạo ra từ nhiều góc quét (multi-view scanning), chúng cần được căn chỉnh để tạo ra một tập hợp hoàn chỉnh. Sai lệch căn chỉnh xảy ra khi các tập dữ liệu không được ghép nối chính xác.

**Nguyên nhân**:

- Thiết bị định vị không chính xác (ví dụ: cảm biến đo góc có sai số).
- Dữ liệu bị thiếu (các vùng trống làm khó khăn cho việc ghép nối).

**Hệ quả**: Các điểm từ các góc nhìn khác nhau không khớp với nhau, tạo ra các méo mó hoặc khoảng trống trên bề mặt tái dựng.

**Giải pháp thường dùng**:

1. **Alignment algorithms**: Các thuật toán căn chỉnh như ICP (Iterative Closest Point).
2. **Use of markers**: Đặt các dấu mốc cố định trong không gian để làm điểm tham chiếu.

---

### **5. Thiếu điểm (Missing points)**

**Ý nghĩa**: Một số vùng trên bề mặt không được quét đủ điểm hoặc không có điểm nào.

**Nguyên nhân**:

- Góc nhìn hạn chế: Máy quét không thể nhìn thấy toàn bộ bề mặt.
- Phản xạ ánh sáng: Các bề mặt quá bóng hoặc trong suốt không trả lại dữ liệu đo.

**Hệ quả**: Các vùng bị thiếu sẽ tạo ra các "lỗ" trong bề mặt tái dựng.

**Ví dụ**: Khi quét một chiếc ly thủy tinh, phần trong suốt của ly không được ghi nhận đầy đủ.

**Giải pháp thường dùng**:

1. **Hole-filling algorithms**: Các thuật toán nội suy để lấp đầy vùng trống.
2. **Multi-view scanning**: Sử dụng nhiều góc quét để giảm thiểu các vùng bị thiếu.\

---

### **Dataset**

Dưới đây là bảng so sánh chi tiết các bộ dữ liệu (datasets) và benchmark được mô tả trong bài báo. Mỗi yếu tố bao gồm thông tin về loại dữ liệu, nguồn, đặc điểm nổi bật, hạn chế, số lượng mẫu, và số lượng lớp.

| **Tên Dataset** | **Loại Dữ Liệu** | **Nguồn (Source)** | **Đặc điểm nổi bật** | **Hạn chế** | **Số lượng mẫu** | **Số lượng lớp** |
| --- | --- | --- | --- | --- | --- | --- |
| **ShapeNet [18]** | Synthetic (object-level) | Tổng hợp từ nhiều nguồn | Gồm các hình khối đơn giản, phù hợp để nghiên cứu các phương pháp cơ bản về tái dựng bề mặt. | Chỉ chứa các hình dạng đơn giản; không đại diện cho các bề mặt phức tạp trong thực tế. | 57,000+ | ~55 (danh mục) |
| **ModelNet [19]** | Synthetic (object-level) | CAD từ các nguồn công nghiệp | Các mẫu CAD tiêu chuẩn, chủ yếu tập trung vào các đối tượng thường gặp như đồ nội thất và vật dụng hàng ngày. | Không bao gồm các lỗi cảm biến thực tế như nhiễu hay lỗ hổng. | 127,000+ | 40/10 (Class/Subset) |
| **3DNet [20]** | Synthetic (object-level) | Thu thập từ CAD công nghiệp | Chứa các đối tượng công nghiệp, gồm các bề mặt phức tạp hơn so với ShapeNet và ModelNet. | Hạn chế về số lượng so với các dataset lớn khác. | ~50,000 | ~200 |
| **ABC [21]** | Synthetic (object-level) | CAD kỹ thuật từ công nghiệp | Các mẫu CAD chính xác, hỗ trợ cả các nghiên cứu về tối ưu hóa lưới tam giác. | Dữ liệu hướng nhiều về CAD kỹ thuật, không phù hợp với các bề mặt tự nhiên hoặc phi công nghiệp. | ~1 triệu | Không phân lớp |
| **Thingi10k [22]** | Synthetic (object-level) | Mẫu in 3D công khai trực tuyến | Được thu thập từ các tệp in 3D, gồm các mẫu đa dạng, có các chi tiết phức tạp và đa vật liệu. | Không chuẩn hóa về chất lượng dữ liệu; chứa các bề mặt không khép kín (non-watertight). | 10,000+ | Không phân lớp |
| **Three D Scans [23]** | Synthetic (object-level) | Tác phẩm điêu khắc nghệ thuật | Gồm các bề mặt phức tạp của các tác phẩm điêu khắc, hỗ trợ các bài toán phức tạp về tái dựng hình dạng. | Số lượng mẫu hạn chế so với các dataset khác. | 106 | Không phân lớp |
| **SceneNet [24]** | Synthetic (scene-level) | Phòng ảo mô phỏng | Dữ liệu cảnh trong nhà với thông tin về hình học và vật liệu, hỗ trợ nghiên cứu tái dựng bề mặt cảnh lớn. | Không tính đến lỗi quét thực tế (nhiễu, lỗ hổng, v.v.). | ~20,000 | Không phân lớp |
| **3D-FRONT [25]** | Synthetic (scene-level) | Dữ liệu phòng ảo công nghiệp | Cảnh nội thất được thiết kế chi tiết với nhiều loại phòng và đồ vật. | Không bao gồm dữ liệu cảm biến thực tế. | 10,000+ | Không phân lớp |
| **Replica [29]** | Real-scanned (scene-level) | Dữ liệu quét thực tế | Cảnh trong nhà được quét thực tế, hỗ trợ nghiên cứu tái dựng bề mặt từ dữ liệu thực tế. | Không có ground truth chất lượng cao do hạn chế của thiết bị quét. | ~50 | Không phân lớp |
| **Tác giả** | Synthetic và Real-scanned | Tổng hợp mới | Dữ liệu đa dạng, gồm các mẫu synthetic và real-scanned; mô phỏng đầy đủ các lỗi quét thực tế như nhiễu, lỗ hổng, sai lệch. | Số lượng cảnh lớn nhưng vẫn có thể mở rộng thêm để bao phủ nhiều trường hợp thực tế khác. | 1,620 object; 50 scene | Không phân lớp |

---

### **Giải thích chi tiết về các yếu tố**

1. **Loại dữ liệu**: Các dataset có thể thuộc hai loại chính:
    - **Synthetic**: Dữ liệu nhân tạo được tạo từ CAD hoặc các mô hình thiết kế sẵn.
    - **Real-scanned**: Dữ liệu thu thập bằng cách quét thực tế từ cảm biến.
2. **Nguồn (source)**:
    - Dữ liệu synthetic thường được lấy từ các bộ sưu tập CAD, các cộng đồng in 3D, hoặc từ các công cụ tạo mô phỏng.
    - Dữ liệu real-scanned thường được thu thập qua các cảm biến 3D như lidar hoặc máy quét quang học.
3. **Đặc điểm nổi bật**:
    - Các dataset như ShapeNet và ModelNet tập trung vào các đối tượng cơ bản để phát triển các thuật toán tái dựng nền tảng.
    - Dataset như ABC hoặc Three D Scans cung cấp các đối tượng phức tạp hơn, phù hợp để nghiên cứu các bài toán nâng cao.
    - Bộ dữ liệu của bài báo đóng góp một benchmark toàn diện, chứa cả synthetic và real-scanned, có thể kiểm tra các thuật toán dưới các lỗi thực tế.
4. **Hạn chế**:
    - Nhiều dataset chỉ chứa dữ liệu synthetic, không phản ánh các vấn đề thực tế (như nhiễu và lỗ hổng).
    - Các dataset real-scanned thường không có ground truth chính xác do hạn chế của thiết bị.
5. **Số lượng**:
    - Số lượng mẫu có sự chênh lệch lớn, từ vài chục (Three D Scans, Replica) đến hàng trăm nghìn mẫu (ModelNet, ShapeNet).
6. **Class**:
    - Một số dataset (ShapeNet, ModelNet) cung cấp thông tin phân lớp, giúp nghiên cứu các bài toán liên quan đến phân loại đối tượng.

---

### **Explicit & Implicit**

***Explicit*** ám chỉ sự rõ ràng, thẳng thắn trong khi ***implicit*** lại hàm ý những biểu thị ngấm ngầm, không rõ ràng.

[An Explication on the Use of 'Explicit' and 'Implicit'](https://www.merriam-webster.com/grammar/usage-of-explicit-vs-implicit)

| Đặc điểm | Explicit Representation | Implicit Representation |
| --- | --- | --- |
| **Biểu diễn bề mặt** | Qua hàm tham số f(x). | Qua hàm implicit F(q) = 0. |
| **Dễ sinh điểm** | Dễ dàng tính điểm từ f(x). | Phải giải phương trình F(q) = 0. |
| **Khả năng tổng quát** | Hạn chế với bề mặt phức tạp. | Phù hợp với mọi bề mặt. |
| **Độ linh hoạt** | Cần miền tham số rõ ràng. | Không yêu cầu miền tham số. |

### **Explicit Representation (Biểu diễn tường minh)**

Đây là cách biểu diễn surface bằng một **hàm tham số (parametric mapping)** cụ thể, cho phép ta trực tiếp xác định được các điểm trên bề mặt bằng cách áp dụng công thức hoặc hàm số vào một không gian đầu vào. Bề mặt $S$ được biểu diễn như một hàm:

$$
f : \Omega^2 \to S, \quad S = \{f(x) \in \mathbb{R}^3 \mid x \in \Omega^2\}
$$

- $\Omega^2 \subset \mathbb{R}^2$: Miền tham số 2D đại diện cho không gian đầu vào.
- $f(x)$: Một hàm ánh xạ từ miền tham số $\Omega^2$ tới các tọa độ 3D trên bề mặt $S$.
- **Ý nghĩa**: Trong biểu diễn tường minh, ta biết rõ cách tạo ra surface $S$ bằng công thức
- **Ví dụ**: Một hình cầu có thể được biểu diễn tường minh bằng:
    
    $$
    f(u, v) = \left(R \cos u \sin v, R \sin u \sin v, R \cos v\right), \quad u \in [0, 2\pi], \; v \in [0, \pi]
    $$
    
    Trong đó $u, v$ là tham số thể hiện vị trí trên bề mặt hình cầu.
    
- **Ưu điểm**:
    - Có thể dễ dàng truy cập và tạo dựng surface $S$ qua công thức cụ thể.
    - Dễ dàng hiển thị hoặc tính toán các điểm trên bề mặt.
- **Hạn chế**:
    - Không phù hợp khi bề mặt phức tạp hoặc có dạng hình học không dễ mô tả bằng công thức.
    - Đôi khi yêu cầu miền tham số $\Omega^2$ phải được xác định trước.

### **Implicit Representation (Biểu diễn ngầm định)**

Thay vì xác định tường minh các point trên surface, biểu diễn ngầm định định nghĩa bề mặt qua một **hàm ngầm định (implicit function)**. Cụ thể, một bề mặt $S$ được mô tả như **zero-level set** của một hàm $F: \mathbb{R}^3 \to \mathbb{R}$:

$$
S = \{q \in \mathbb{R}^3 \mid F(q) = 0\}
$$

- **Ý nghĩa toán học**: Hàm $F(q)$ trả về một giá trị thể hiện mối quan hệ của điểm $q$ với bề mặt:
    - Nếu $F(q) = 0$:  nằm trên bề mặt.
    - Nếu $F(q) > 0$:  nằm ngoài bề mặt.
    - Nếu $F(q)<0$:  nằm trong bề mặt.
- **Ví dụ**: Một hình cầu có tâm tại $(0,0,0)$  và bán kính $R$ có thể được biểu diễn ngầm định bởi:
    
    $$
    F(q) = \|q\|^2 - R^2, \quad S = \{q \in \mathbb{R}^3 \mid F(q) = 0\}
    $$
    
- **Ưu điểm**:
    - Không yêu cầu miền tham số rõ ràng, rất phù hợp để mô tả bề mặt phức tạp hoặc không đều.
    - Có thể được áp dụng để biểu diễn cả các bề mặt kín và mở.
    - Linh hoạt hơn trong các bài toán tối ưu hóa vì chỉ cần đánh giá hàm $F(q)$.
- **Hạn chế**:
    - Không dễ dàng sinh trực tiếp tọa độ các điểm trên bề mặt như biểu diễn tường minh.
    - Có thể yêu cầu nhiều tính toán để trích xuất bề mặt từ hàm $F(q)$, ví dụ như sử dụng thuật toán Marching Cubes.

---

### **Hypothesis Space**

**Hypothesis space ($H_f$ hoặc $H_F$)** là tập hợp các hàm khả dĩ $f$ (hoặc $F$) mà mô hình có thể chọn để biểu diễn bề mặt.

- Với explicit representation: $H_f$ có thể là tập các hàm tham số hóa bề mặt (như B-spline, NURBS, hoặc mạng neuron).
- Với implicit representation:  $H_F$ là tập các hàm ngầm định (như Signed Distance Function hoặc Occupancy Field).

**Vai trò**: Hypothesis space định nghĩa "vùng tìm kiếm" cho bề mặt phù hợp nhất .

**Ví dụ**: Nếu $H_f$ được định nghĩa qua mạng neuron $f(x; \theta)$, thì bài toán tối ưu hóa là tìm tham số  $\theta$ để $f$ khớp dữ liệu $P$.

---

### **Data Fidelity**

[Fidelity - MOSTLY AI](https://mostly.ai/synthetic-data-dictionary/fidelity)

Data fidelity (độ trung thực dữ liệu) là một thành phần trong hàm mục tiêu để đảm bảo rằng surface reconstruction $S$ phù hợp với dữ liệu quan sát được $P$. Nó đo mức độ surface $S$ khớp với input $P$ bằng cách sử dụng một hàm loss $L(S;P)$.

- Đảm bảo rằng các điểm $p \in P$ nằm gần bề mặt $S$ được tái dựng.
- Ví dụ: Nếu bề mặt $S$ quá xa so với các điểm $P$, giá trị của hàm loss $L$ sẽ rất lớn, chỉ ra rằng bề mặt không phản ánh đúng dữ liệu gốc.

**Ví dụ cụ thể trong bài toán** với ∥⋅∥ là một loại khoảng cách (ví dụ: khoảng cách Euclide).

$$
L(S; P) = \frac{1}{n_P} \sum_{p \in P} \min_{x \in \Omega^2} \|f(x) - p\|_\ell,
$$

---

### **Signed Distance Function (SDF) & Occupancy Field**

[Signed distance fields](https://jasmcole.com/2019/10/03/signed-distance-fields/)

[Inigo Quilez](https://iquilezles.org/articles/distfunctions/)

### **Signed Distance Function (SDF)**

- **Signed Distance Function** là một hàm ngầm định $F(q)$ dùng để biểu diễn bề mặt 3D. Nó được định nghĩa dựa trên khoảng cách có hướng giữa một điểm trong không gian $q$ và bề mặt cần tái dựng $S$.
- Cụ thể:
    - $F(q) > 0$: Khi $q$ nằm **bên ngoài** bề mặt.
    - $F(q) < 0$: Khi $q$ nằm **bên trong** bề mặt.
    - $F(q) = 0$: Khi $q$ nằm **trên bề mặt**.

### **Ý nghĩa trong tái dựng bề mặt**:

1. **Định nghĩa bề mặt rõ ràng**: SDF biểu diễn bề mặt dưới dạng tập hợp mức (level set) $F(q) = 0$, giúp mô tả hình dạng bề mặt một cách ngầm định mà không cần phải lưu trữ toàn bộ dữ liệu điểm.
2. **Tái dựng chính xác**: Vì SDF dựa trên khoảng cách thực, nó rất chính xác trong việc tái dựng hình học mịn màng và phức tạp.
3. **Hỗ trợ gradient**: Gradient của $F(q)$ tại điểm $q$ cung cấp thông tin về hướng và độ dốc của bề mặt, giúp cải thiện việc tối ưu hóa trong học sâu.

**Ví dụ**: Một quả cầu với tâm  $c$ và bán kính $r$ có thể được biểu diễn bởi $F(q) = \|q - c\| - r$. Giá trị này là dương nếu $q$ nằm ngoài quả cầu, âm nếu bên trong, và bằng 0 nếu trên bề mặt quả cầu.

**Dẫn chứng từ bài báo**: Bài báo chỉ rõ rằng SDF được sử dụng để tính khoảng cách ký hiệu giữa điểm $q$ và tập điểm đã quét $P$. Điều này rất quan trọng để đảm bảo $q$ khớp chính xác với bề mặt.

---

### **Occupancy Field (OF)**

- **Occupancy Field** là một hàm ngầm định khác $F(q)$ được sử dụng để kiểm tra xem một điểm $q$ nằm **bên trong hay bên ngoài** bề mặt.
- Cụ thể:
    - $F(q) = 1$: Nếu $q$ nằm bên trong bề mặt $S$
    - $F(q) = 0$: Nếu $q$ nằm bên ngoài bề mặt $S$.

### **Ý nghĩa trong tái dựng bề mặt**:

1. **Phân vùng không gian**: OF chia không gian 3D thành hai vùng rõ ràng: bên trong và bên ngoài bề mặt, giúp xác định cấu trúc bề mặt một cách hiệu quả.
2. **Hỗ trợ tái dựng từ dữ liệu không hoàn chỉnh**: OF đặc biệt hữu ích khi dữ liệu quét bị thiếu điểm (missing points), vì nó không yêu cầu thông tin chính xác về khoảng cách.
3. Deep learning: Trong các phương pháp dựa trên deep learninng, như Occupancy Networks, OF thường được học từ dữ liệu và sử dụng để tái dựng bề mặt bằng cách tối ưu hóa một hàm phi tuyến.

**Ví dụ**: Hãy tưởng tượng bạn có một hình hộp chữ nhật với các đỉnh $a$ và $b$. Một điểm $q$ nằm bên trong hộp sẽ có $F(q) = 1$, trong khi $q$ nằm ngoài sẽ có $F(q) = 0$.

**Dẫn chứng từ bài báo**: Bài báo nhấn mạnh rằng OF được ước tính bằng cách so sánh vị trí của $q$ với tập dữ liệu điểm $P$, cung cấp cách tiếp cận nhị phân đơn giản nhưng hiệu quả cho bài toán tái dựng bề mặt

---

### **Local Tangent Plane, Normal Vector**

[Why is the gradient related to the normal vector to a surface?](https://samjshah.com/2009/01/16/why-is-the-gradient-related-to-the-normal-vector-to-a-surface/)

[Unit 2: Tangent Plane](https://pressbooks.library.torontomu.ca/multivariatecalculus/chapter/unit-2-tangent-plane/)

**Local Tangent Plane** là mặt phẳng tiếp xúc với bề mặt $S$ tại một điểm $p$. Nó được định nghĩa bởi:

1. **Điểm $p$** trên bề mặt $S$.
2. **Pháp tuyến** $n$ của bề mặt tại $p$, chỉ hướng vuông góc với mặt phẳng.

### **Cách tính toán**:

1. **Tìm lân cận**: Xác định các điểm lân cận  $N(p)$ xung quanh $p$ từ tập dữ liệu $P$ (thường sử dụng k-nearest).
2. **Phân tích thành phần chính (PCA)**: Sử dụng PCA trên các điểm $N(p)$ để tìm vector chính đầu tiên (vector pháp tuyến) và các vector còn lại tạo nên mặt phẳng tiếp tuyến.
3. **Tạo mặt phẳng**: Mặt phẳng được xác định bởi vector pháp tuyến và $p$.

### **Ý nghĩa trong tái dựng bề mặt**:

1. **Hỗ trợ tính pháp tuyến**: Mặt phẳng tiếp tuyến cung cấp pháp tuyến $n$, giúp định hướng bề mặt trong không gian 3D.
2. **Mịn hóa bề mặt**: Thông tin từ mặt phẳng tiếp tuyến giúp tái dựng các bề mặt mịn màng, đặc biệt là trong các phương pháp dựa trên SDF.
3. **Chuyển đổi tọa độ**: Mặt phẳng tiếp tuyến được sử dụng để tạo hệ tọa độ địa phương xung quanh mỗi điểm $p$, giúp phân tích hình học tốt hơn.

**Ví dụ**: Xét một bề mặt phẳng trong không gian. Tại một điểm $p$, tất cả các điểm $q$ nằm trong mặt phẳng này sẽ thỏa mãn phương trình:

$$
(q - p) \cdot n = 0,
$$

trong đó $n$ là pháp tuyến của bề mặt tại $p$.

### **Dẫn chứng từ bài báo**:

Bài báo sử dụng khái niệm "local tangent plane" để mô tả cách tính pháp tuyến $n(q; P)$, một thành phần quan trọng trong việc tính toán các độ đo như gradient của SDF và OF.

---

### **Gradient**

### **1. Gradient là gì?**

### **1.1. Định nghĩa cơ bản**

Gradient là một khái niệm trong toán học, dùng để biểu diễn **hướng thay đổi lớn nhất** của một hàm tại một điểm.

Để dễ hiểu, hãy tưởng tượng bạn đang leo một ngọn đồi. Hàm số $F(q)$ biểu diễn độ cao của ngọn đồi tại mỗi điểm $q$ trên mặt đất. Gradient của $F(q)$, ký hiệu $\nabla_q F(q)$, là một vector:

- Chỉ **hướng dốc nhất** để leo lên ngọn đồi (hướng tăng nhanh nhất của độ cao).
- **Độ lớn** của gradient thể hiện độ dốc: Gradient càng lớn, ngọn đồi càng dốc.

### **1.2. Công thức tính gradient**

Trong không gian 3 chiều $\mathbb{R}^3$, một hàm số $F(q)$ có thể phụ thuộc vào ba biến $x, y, z$. Gradient của $F(q)$ được tính bằng:

$$
\nabla_q F(q) = \left( \frac{\partial F}{\partial x}, \frac{\partial F}{\partial y}, \frac{\partial F}{\partial z} \right)
$$

Trong đó:

- $\frac{\partial F}{\partial x}$: Tốc độ thay đổi của $F(q)$ theo hướng $x$.
- $\frac{\partial F}{\partial y}$: Tốc độ thay đổi theo hướng $y$.
- $\frac{\partial F}{\partial z}$: Tốc độ thay đổi theo hướng $z$

Gradient là một vector ba chiều chứa thông tin về **hướng** và **độ lớn** của sự thay đổi của $F$ tại điểm $q$.

---

### **2. Ý nghĩa của Gradient trong Surface Reconstruction**

Trong bài toán tái tạo bề mặt, hàm ngầm $F(q)$ được dùng để mô hình hóa bề mặt 3D:

- $F(q) = 0$: Điểm $q$ nằm trên bề mặt.
- $F(q) > 0$: Điểm $q$ nằm bên ngoài bề mặt.
- $F(q) > 0$: Điểm $q$ nằm bên trong bề mặt.

### **Vai trò của Gradient trong bài toán**

Gradient $\nabla_q F(q)$ đóng vai trò:

1. **Chỉ hướng pháp tuyến (normal direction)**: Hướng vuông góc với bề mặt tại điểm $q$.
2. **Mô tả hình dạng bề mặt**: Gradient cho biết bề mặt dốc như thế nào xung quanh điểm $q$

---

### **3. Thành phần Gradient trong Hàm Loss**

Hàm mất mát nâng cao $\mathcal{L}_{\text{imp++}}(F; P)$ chứa một thành phần để khớp gradient $\nabla_q F(q)$ với vector pháp tuyến $n(q; P)$:

$$
\mathbb{E}_{q \in \mathbb{R}^3} \|\nabla_q F(q) - n(q; P)\|_2
$$

### **3.1. Thành phần Gradient**

- $\nabla_q F(q)$: Gradient của hàm $F$ tại điểm $q$. Đây là vector pháp tuyến ước lượng từ hàm $F$, chỉ hướng ra ngoài bề mặt $S$.
- $n(q; P)$: Vector pháp tuyến thực tại điểm $q$, tính toán từ tập dữ liệu $P$ (các điểm lân cận $q$).

### **3.2. Ý nghĩa của thành phần này**

- Thành phần này đo độ sai lệch giữa pháp tuyến ước lượng $\nabla_q F(q)$ và pháp tuyến thực $n(q; P)$.
- Mục tiêu là:
    
    $$
    \nabla_q F(q) \approx n(q; P)
    $$
    
    Điều này đảm bảo:
    
    1. $F(q)$ không chỉ mô hình hóa chính xác hình dạng bề mặt mà còn khớp với hướng pháp tuyến.
    2. Bề mặt được tái tạo trơn tru và chính xác hơn, đặc biệt ở các khu vực cong hoặc có góc cạnh.

---

### **4. Cách tính pháp tuyến thực n(q; P) từ dữ liệu**

Để tính $n(q; P)$, cần xác định mặt phẳng tiếp tuyến (tangent plane) tại điểm $q$. Đây là cách làm:

### **4.1. Thu thập các điểm lân cận**

- Chọn các điểm gần $q$ nhất trong tập dữ liệu $P$, gọi là $\{p_1, p_2, \ldots, p_k\}$. Đây là các điểm lân cận gần $q$ theo khoảng cách Euclidean

### **4.2. Phân tích Thành phần Chính (PCA)**

- Sử dụng Phân tích Thành phần Chính (Principal Component Analysis, PCA) để tìm mặt phẳng tiếp tuyến. PCA xác định hướng chính trong dữ liệu lân cận, từ đó tính ra:
    - **Pháp tuyến** $n(q; P)$: Vector vuông góc với mặt phẳng tiếp tuyến.
    - **Các hướng chính**: Song song với bề mặt.

### **4.3. Chuẩn hóa pháp tuyến**

Đảm bảo pháp tuyến có độ dài bằng 1:

$$
n(q; P) = \frac{n(q; P)}{\|n(q; P)\|}
$$

---

### **5. Vai trò Toàn Diện của Gradient trong Surface Reconstruction**

### **5.1. Mối liên hệ Gradient - Pháp tuyến**

- Gradient $\nabla_q F(q)$ không chỉ mô tả sự thay đổi của hàm $F$
, mà còn trực tiếp biểu diễn vector pháp tuyến tại điểm  $q$ khi:
    
    $$
    F(q) = 0
    $$
    
    - Đây là đặc điểm quan trọng của các hàm implicit như **Signed Distance Function (SDF)**.

### **5.2. Trong Hàm Mất Mát $\mathcal{L}_{\text{imp++}}(F; P)$**

Thành phần gradient đảm bảo rằng mô hình không chỉ khớp hình dạng bề mặt mà còn đảm bảo tính chính xác của pháp tuyến, giúp:

- Tái tạo bề mặt mượt mà.
- Khớp các góc cạnh và chi tiết hình học.

### **5.3. Kết hợp với các thành phần khác**

Gradient phối hợp với thành phần chính của $\mathcal{L}_{\text{imp++}}(F; P)$, như $\|F(q) - d(q; P)\|_1$, để mô hình hóa toàn bộ hình dạng bề mặt và các tính chất cục bộ như độ mịn và hướng.

---

### **Mesh 3d ? Type of Mesh ?**

[What is 3D mesh? | Definition from TechTarget](https://www.techtarget.com/whatis/definition/3D-mesh)

[Types of mesh](https://en.wikipedia.org/wiki/Types_of_mesh)

---

### **Circumscribing Sphere of a Triangle**

[Circumscribed sphere](https://en.wikipedia.org/wiki/Circumscribed_sphere)

Mặt cầu ngoại tiếp tam giác là mặt cầu đi qua tất cả ba đỉnh của một tam giác $T$.

- Mặt cầu này là duy nhất cho một tam giác không suy biến (tam giác có diện tích khác 0).
- Tâm và bán kính của mặt cầu được xác định dựa trên các đỉnh của tam giác.

**Tính chất quan trọng trong Delaunay Triangulation**:

Điều kiện **empty sphere property** yêu cầu rằng không có điểm nào khác trong tập dữ liệu $P$ nằm bên trong mặt cầu ngoại tiếp của tam giác $T$. Điều này đảm bảo tam giác được chọn phù hợp về mặt hình học, giảm thiểu các tam giác "xấu".

**Ví dụ minh họa**: Cho một tam giác $T$ với ba đỉnh $A(0, 0), B(1, 0), C(0, 1)$:

- Mặt cầu ngoại tiếp T có tâm $O = (0.5, 0.5)$ và bán kính $r = \sqrt{0.5^2 + 0.5^2} = \sqrt{0.5}$
- Nếu có một điểm  $D(0.3,0.4)$ nằm bên trong mặt cầu, tam giác này không được chọn trong Delaunay Triangulation.

---

### **Locally Differentiable**

![image.png](Surface%20Reconstruction%20From%20Point%20Clouds%20A%20Survey%20%201442a3f8f2a180aca990fd4b2793fb3b/image%2021.png)

Một bề mặt được gọi là "trơn cục bộ" nếu nó khả vi tại tất cả các điểm bên trong một vùng cục bộ, và không có sự "đứt gãy" hoặc "nhọn" trong vùng đó.

→ Nói cách khác, bề mặt tại một vùng nhỏ có thể được mô tả bằng các đoạn trơn, liên tục, không có góc nhọn.

**Ý nghĩa trong bài toán Surface Reconstruction**: Giả thiết bề mặt trơn cục bộ cho phép sử dụng các tam giác phẳng nhỏ để xấp xỉ bề mặt trong từng phần nhỏ.

![image.png](Surface%20Reconstruction%20From%20Point%20Clouds%20A%20Survey%20%201442a3f8f2a180aca990fd4b2793fb3b/image%2022.png)

**Ví dụ minh họa**: Bề mặt của một quả bóng cao su là một ví dụ điển hình về bề mặt trơn cục bộ. Ở mỗi vùng nhỏ, ta có thể xấp xỉ nó bằng các tam giác phẳng mà không làm mất đi độ mượt.

### **Locally Differentiable up to First Order**

- Một hàm hoặc bề mặt $S$ được gọi là **khả vi bậc 1 (first-order differentiable)** nếu tại mỗi điểm x, nó có **đạo hàm** xác định (cả về hướng và giá trị).
- Nói cách khác, tại mỗi điểm x, ta có thể tìm một mặt phẳng tiếp tuyến (tangent plane) cục bộ, mà bề mặt xung quanh điểm đó có thể được xấp xỉ tuyến tính bởi mặt phẳng này.

**Ý nghĩa trong bài toán Surface Reconstruction**:

- Tính khả vi bậc 1 giúp bề mặt có thể được xấp xỉ bằng các tam giác phẳng nhỏ cục bộ (xem thuật ngữ **Piecewise Linear** bên dưới).
- Điều này đảm bảo rằng bề mặt không bị "đứt gãy" hoặc có góc nhọn bất thường.

**Ví dụ minh họa**: Xét đồ thị của hàm $z = x^2 + y^2$:

- Tại điểm $(1, 1, 2)$, đạo hàm bậc 1 tồn tại và cho phép ta xác định một mặt phẳng tiếp tuyến  $z=2+2x+2y.$
- Điều này có nghĩa rằng quanh điểm $(1, 1)$, surface có thể được xấp xỉ bởi mặt phẳng này.

---

### **Piecewise Linear**

[Piecewise Linear Function -- from Wolfram MathWorld](https://mathworld.wolfram.com/PiecewiseLinearFunction.html)

![image.png](Surface%20Reconstruction%20From%20Point%20Clouds%20A%20Survey%20%201442a3f8f2a180aca990fd4b2793fb3b/image%2023.png)

- **Piecewise linear** là cách biểu diễn một bề mặt hoặc một hàm bằng cách ghép nhiều đoạn tuyến tính lại với nhau.
- Mỗi đoạn tuyến tính đại diện cho một phần nhỏ của bề mặt hoặc không gian.

**Ý nghĩa trong bài toán Surface Reconstruction**:

Bề mặt S được biểu diễn như một tập hợp các tam giác phẳng:

$$
S = \bigcup_{i=1}^{n_G} T_i
$$

Trong đó, $T_i$ là các tam giác tạo nên lưới tam giác.

**Ví dụ minh họa**: Hãy tưởng tượng bề mặt một ngọn núi được chia thành các mảnh nhỏ (mỗi mảnh là một tam giác phẳng). Mặc dù ngọn núi thực sự không phẳng, nhưng các tam giác này có thể xấp xỉ khá tốt hình dạng tổng thể.

---

### **Topological Constraints**

![image.png](Surface%20Reconstruction%20From%20Point%20Clouds%20A%20Survey%20%201442a3f8f2a180aca990fd4b2793fb3b/image%2024.png)

**Tô-pô (Topology)** là lĩnh vực nghiên cứu các tính chất không đổi của một không gian khi bị biến dạng liên tục, chẳng hạn như số lỗ hổng hoặc thành phần kết nối.

**Topological constraints** trong bài toán tái tạo bề mặt là các ràng buộc đảm bảo rằng bề mặt tái tạo có cùng cấu trúc tô-pô như dữ liệu quan sát.

**Ý nghĩa trong bài toán Surface Reconstruction**: Những ràng buộc này đảm bảo rằng bề mặt tái tạo không xuất hiện các lỗ hổng không mong muốn hoặc không kết nối các thành phần khác nhau một cách bất thường.

**Ví dụ minh họa**: Dữ liệu của một hình cầu có một thành phần kết nối duy nhất và không có lỗ. Topological constraints đảm bảo rằng bề mặt tái tạo không xuất hiện lỗ hoặc bị tách rời.

---

### Radial Basis Function (RBF)

[Radial Basis Functions: Types, Advantages, and Use Cases | HackerNoon](https://hackernoon.com/radial-basis-functions-types-advantages-and-use-cases)

### **1. Radial Basis Function là gì?**

Radial Basis Function (RBF) là một loại **hàm toán học** được sử dụng phổ biến trong các bài toán học máy (machine learning) và nội suy dữ liệu (interpolation). Một RBF phụ thuộc vào **khoảng cách** giữa hai điểm trong không gian, thông thường khoảng cách này được đo bằng **norm Euclidean**.

Một RBF có dạng tổng quát:

$$
\phi(r) = \phi(\|x - c\|)
$$

Trong đó:

- $\|x - c\|$: Khoảng cách giữa hai điểm $x$ và $c$ (center of the basis function).
- $r = \|x - c\|$: Giá trị khoảng cách.
- $\phi(r)$: Hàm cơ sở, phụ thuộc chỉ vào giá trị $r$.

---

### **2. Các loại Radial Basis Function phổ biến**

Một số hàm RBF thông dụng bao gồm:

1. **Gaussian RBF**:
    
    $$
    \phi(r) = \exp\left(-\frac{r^2}{2\sigma^2}\right)
    $$
    
    - **Ý nghĩa**: Hàm này giảm dần giá trị khi khoảng cách $r$ tăng lên, và được kiểm soát bởi tham số $\sigma$ (điều chỉnh mức độ ảnh hưởng).
    - **Ví dụ**: Gaussian RBF thường được sử dụng trong các mô hình như SVM (Support Vector Machines) với kernel RBF.
2. **Multiquadric RBF**:
    
    $$
    \phi(r) = \sqrt{r^2 + \epsilon^2}
    $$
    
    - $\epsilon$: Tham số điều chỉnh để tránh giá trị bằng 0 khi $r = 0$.
    - **Ứng dụng**: Dùng trong nội suy để xử lý các bề mặt không đồng đều.
3. **Inverse Multiquadric RBF**:
    
    $$
    \phi(r) = \frac{1}{\sqrt{r^2 + \epsilon^2}}
    $$
    
4. **Thin Plate Spline RBF**:
    
    $$
    \phi(r) = r^2 \log(r)
    $$
    
    - **Ứng dụng**: Dùng để mô hình hóa bề mặt mịn trong các bài toán đồ họa máy tính.

---

### **3. Ứng dụng trong Surface Reconstruction**

Trong bài toán tái tạo bề mặt (surface reconstruction), RBF được sử dụng để làm mịn các dữ liệu đầu vào (point cloud). Ý tưởng là mỗi điểm trong đám mây điểm sẽ có ảnh hưởng lên các điểm lân cận của nó, với mức độ ảnh hưởng giảm dần theo khoảng cách.

### **Ví dụ cụ thể**:

Giả sử chúng ta có một tập điểm $P = \{p_1, p_2, \ldots, p_N\}$ trong không gian 3D. Để làm mịn một điểm $p$, ta tính giá trị trung bình có trọng số từ các điểm lân cận dựa trên RBF:

$$
\hat{p}(p) = \sum_{p' \in N(p)} p' \cdot g(\|p - p'\|)
$$

Trong đó:

- $N(p)$: Tập hợp các điểm lân cận của $p$.
- $g(\|p - p'\|)$: Trọng số được xác định bởi một RBF, ví dụ:
    
    $$
    g(\|p - p'\|) = \exp\left(-\frac{\|p - p'\|^2}{2\sigma^2}\right)
    $$
    

**Ý nghĩa**:

- Điểm $p'$càng gần $p$ thì trọng số $g(\|p - p'\|)$ càng lớn.
- Điểm $p'$ càng xa $p$ thì trọng số $g(\|p - p'\|)$ giảm nhanh chóng, làm giảm ảnh hưởng của điểm này.

---

### **4. Minh họa đơn giản**

Hãy tưởng tượng một ví dụ trong không gian 2D, bạn có một số điểm dữ liệu rời rạc như sau:

| x | y | Giá trị (hàm cần nội suy) |
| --- | --- | --- |
| 1 | 1 | 2 |
| 2 | 3 | 3 |
| 4 | 5 | 4 |

Giả sử bạn muốn ước lượng giá trị tại $x = 3, y = 4$. Sử dụng Gaussian RBF với $\sigma = 1.0$, trọng số của từng điểm dữ liệu được tính như sau:

$$
w_i = \exp\left(-\frac{\|p - p_i\|^2}{2\sigma^2}\right)
$$

- $p = (3, 4)$: Điểm cần nội suy.
- $p_1 = (1, 1), p_2 = (2, 3), p_3 = (4, 5)$: Các điểm dữ liệu.

Tính khoảng cách Euclidean:

$$
\|p - p_1\| = \sqrt{(3 - 1)^2 + (4 - 1)^2} = \sqrt{13}
$$

$$
\|p - p_2\| = \sqrt{(3 - 2)^2 + (4 - 3)^2} = \sqrt{2}
$$

$$
\|p - p_3\| = \sqrt{(3 - 4)^2 + (4 - 5)^2} = \sqrt{2}
$$

Tính trọng số Gaussian:

$$
w_1 = \exp\left(-\frac{\sqrt{13}^2}{2}\right), \quad w_2 = \exp\left(-\frac{\sqrt{2}^2}{2}\right), \quad w_3 = \exp\left(-\frac{\sqrt{2}^2}{2}\right)
$$

Giá trị nội suy tại $p$ được tính:

$$
\text{Interpolated Value} = \frac{\sum_{i=1}^3 w_i \cdot y_i}{\sum_{i=1}^3 w_i}
$$

---

### **5. Ý nghĩa thực tiễn**

RBF rất hiệu quả trong việc xử lý dữ liệu không đồng đều hoặc bị nhiễu:

- **Học máy (Machine Learning)**: Dùng trong các kernel như **Gaussian RBF kernel** để phân loại hoặc hồi quy.
- **Tái tạo bề mặt (Surface Reconstruction)**: Làm mịn bề mặt từ dữ liệu đám mây điểm.
- **Đồ họa máy tính (Computer Graphics)**: Nội suy bề mặt hoặc tạo hình ảnh thực tế hơn từ các mẫu dữ liệu thưa thớt.

---

### CAD Model

[CAD (Computer-Aided Design) là gì?](https://onecadvn.com/blog/cad-computer-aided-design-la-gi)

---

### **Non-wateright, Self-occlusion, 2D -manifold,  Surface Genus**

### **1. Non-Watertight Meshes**

- **Định nghĩa**: Một **mesh** được gọi là **watertight** nếu bề mặt của nó được định nghĩa hoàn chỉnh, không có lỗ hổng hoặc khoảng trống. Điều này có nghĩa là tất cả các tam giác trong mesh đều được kết nối chặt chẽ với nhau, tạo thành một khối liền mạch, giống như vỏ kín của một vật thể.
- **Non-Watertight**: Nếu mesh không thỏa mãn điều kiện trên, nó được gọi là **non-watertight**. Các mesh này có thể có các vấn đề như:
    - **Holes**: Các lỗ hổng trên bề mặt, ví dụ: mô hình quả bóng bị rách.
    - **Open edges**: Các cạnh mở không được kết nối với tam giác khác.
- **Ví dụ**:
    - **Watertight**: Một quả cầu hoàn chỉnh với bề mặt kín.
    - **Non-Watertight**: Một ly cà phê mà phần nắp trên cùng bị thiếu.
- **Tầm quan trọng**: Non-watertight meshes có thể gây khó khăn cho các thuật toán tái tạo bề mặt, vì các phương pháp này thường yêu cầu dữ liệu có cấu trúc đầy đủ để tạo bề mặt liền mạch.

---

### **2. Self-Occlusion**

- **Định nghĩa**: **Self-occlusion** xảy ra khi một phần của bề mặt của đối tượng bị che khuất bởi một phần khác của chính nó. Hiện tượng này thường xảy ra ở các đối tượng có hình dạng phức tạp, ví dụ như các vật thể có các nhánh hoặc lỗ hổng sâu.
- **Ví dụ**:
    - Một bức tượng với cánh tay che khuất phần thân là ví dụ điển hình của self-occlusion.
    - Một mô hình bàn với các chân bàn có thể tự che khuất lẫn nhau từ một số góc nhìn.
- **Tầm quan trọng**: Trong việc tái tạo bề mặt từ dữ liệu quét, self-occlusion có thể gây ra mất dữ liệu cục bộ vì các cảm biến không thể "nhìn thấy" tất cả các phần của bề mặt.

---

### **3. 2D-Manifold**

- **Định nghĩa**: Một bề mặt **2D-manifold** là một bề mặt trong không gian 3D mà mọi điểm trên đó đều có một lân cận có thể ánh xạ tới mặt phẳng 2D. Nói cách khác, bề mặt phải có cấu trúc "như 2D" khi xem xét cục bộ, nhưng tồn tại trong không gian 3D.
- **Điều kiện để 2D-manifold**:
    - Mỗi cạnh trong mesh phải thuộc về đúng hai tam giác.
    - Không có lỗ (holes) và không có các giao cắt không mong muốn.
- **Non-2D-Manifold**:
    - Các mesh có các cạnh chia sẻ hơn hai tam giác, tạo ra các điểm giao nhau bất thường.
    - Ví dụ: Các mô hình có cấu trúc phức tạp như các nút giao hình học hoặc nhiều lớp chồng chéo.
- **Ví dụ**:
    - **2D-Manifold**: Một hình cầu trơn tru hoặc một hình xuyến.
    - **Non-2D-Manifold**: Một bề mặt có các điểm giao cắt không tự nhiên, ví dụ như hai tấm phẳng giao nhau tại một đường thẳng.
- **Tầm quan trọng**: Các thuật toán tái tạo bề mặt thường giả định rằng dữ liệu là 2D-manifold để tránh các lỗi hình học phức tạp trong quá trình tái tạo.

---

### **4. Surface Genus**

- **Định nghĩa**: **Surface genus** là một khái niệm trong tô-pô học, thể hiện số lượng "lỗ tay cầm" hoặc "lỗ thông" trên một bề mặt.
    - Genus được sử dụng để đo độ phức tạp tô-pô của bề mặt.
- **Công thức Euler** (một dạng biểu diễn tô-pô):
    
    $$
    \chi = V - E + F
    $$
    
    Với:
    
    - $V$: Số đỉnh (vertices).
    - $E$: Số cạnh (edges).
    - $F$: Số mặt (faces).
    - $\chi$: Đặc trưng Euler của bề mặt.
    
    Genus $g$ được xác định bởi:
    
    $$
    g = \frac{2 - \chi}{2}
    $$
    
    - Đối với các bề mặt kín, genus đo lường số lượng lỗ thông.

- **Ví dụ**:
    - **Genus 0**: Một hình cầu, không có lỗ tay cầm.
    - **Genus 1**: Một hình xuyến (torus) với một lỗ tay cầm.
    - **Genus 2**: Một mô hình có hai lỗ tay cầm, ví dụ như một chiếc bánh pretzel.
- **Tầm quan trọng**: Surface genus giúp xác định độ phức tạp của mô hình. Các đối tượng có genus cao (nhiều lỗ tay cầm) có thể khó xử lý hơn trong tái tạo bề mặt.

---

### **Quy trình lọc dữ liệu trong bài báo**

Tác giả sử dụng các tiêu chí trên để lọc dữ liệu không phù hợp, nhằm đảm bảo chất lượng của tập dữ liệu được sử dụng để tái tạo bề mặt:

1. **Xác định Watertight và Non-Watertight Meshes**:
    - Tác giả kiểm tra trạng thái **watertight** của các mesh. Các mesh non-watertight có lỗ hoặc cạnh mở được loại bỏ vì chúng gây khó khăn trong việc quét và tái tạo bề mặt.
2. **Kiểm tra Self-Occlusion**:
    - Tác giả sử dụng các kỹ thuật kiểm tra **self-occlusion** để đảm bảo rằng các phần tự che khuất không làm giảm chất lượng của tập dữ liệu.
3. **Xác định 2D-Manifold**:
    - Các mesh không phải 2D-manifold (với các cạnh thuộc hơn hai tam giác hoặc các giao cắt bất thường) được loại bỏ vì chúng gây ra lỗi trong các thuật toán tái tạo bề mặt.
4. **Tính toán Surface Genus**:
    - Tác giả tính toán **surface genus** của mỗi đối tượng để loại bỏ các đối tượng có cấu trúc tô-pô quá phức tạp. Các mô hình có genus cao thường không phù hợp để tái tạo bề mặt thực tế.
5. **Chuẩn hóa Meshes**:
    - Sau khi lọc, các mesh được chuẩn hóa bằng cách:
        - Đặt tâm tại gốc tọa độ.
        - Cân chỉnh kích thước để vừa với hình cầu đơn vị.

**Kết quả cuối cùng**:

Từ bộ dữ liệu ban đầu (~10,000 mẫu), sau khi áp dụng quy trình lọc, tác giả thu được **1,620 đối tượng bề mặt** đáp ứng các tiêu chuẩn trên.

---

### **Istrophy**

[Isotropy](https://en.wikipedia.org/wiki/Isotropy)

---

### Algebraic Complexity, Topological Complexity

### **1. Algebraic Complexity (Độ phức tạp đại số)**

### **Định nghĩa**:

- **Algebraic complexity** là một thước đo dùng để đánh giá mức độ phức tạp của bề mặt dựa trên **bậc cao nhất của đa thức** cần thiết để biểu diễn hoặc xấp xỉ các mảnh (patches) cục bộ của bề mặt đó.
- Nó phản ánh độ cong và hình dạng phức tạp của bề mặt.

### **Cách tính toán**:

- Với một bề mặt SS, chia bề mặt thành các mảnh nhỏ PiP_i.
- Mỗi mảnh PiP_i được xấp xỉ bằng một đa thức fi(x,y,z)f_i(x, y, z) với một bậc nhất định dd:
    
    $$
    f_i(x, y, z) = \sum_{k=0}^d \sum_{j=0}^{k} a_{k,j} x^{k-j} y^j z^{d-k}
    $$
    
    - $d$: Bậc của đa thức.
    - $a_{k,j}$: Các hệ số điều chỉnh của đa thức.
- **Algebraic complexity** được đo bằng **bậc cao nhất của các đa thức** trong tập hợp các đa thức $\{f_i\}$ sử dụng để xấp xỉ toàn bộ bề mặt.

### **Ví dụ minh họa**:

1. **Bề mặt đơn giản**: Một mặt phẳng (plane) có thể được biểu diễn bằng một đa thức bậc 1:
    
    $$
    f(x, y, z) = ax + by + cz + d
    $$
    
    => **Algebraic complexity** = 1 (bậc thấp).
    
2. **Bề mặt phức tạp hơn**: Một hình cầu (sphere) được biểu diễn bằng phương trình:
    
    $$
    f(x, y, z) = x^2 + y^2 + z^2 - r^2
    $$
    
    => **Algebraic complexity** = 2 (bậc cao hơn).
    
3. **Bề mặt phức tạp cao**: Một hình dạng tự do hoặc có nhiều cong (như một mảnh của hình torus - bánh vòng) có thể yêu cầu các đa thức bậc cao hơn.

### **Thách thức khi tính toán**:

- Để xấp xỉ bề mặt bằng các đa thức bậc cao, các phương pháp fitting có thể không ổn định khi lỗi xấp xỉ (approximation error) bị hạn chế.
- Trong thực tế, thay vì tính **bậc cao nhất**, tác giả đề xuất cố định bậc $d$ của các đa thức và đo **trung bình lỗi xấp xỉ (averaged approximation errors)** trên các mảnh cục bộ của bề mặt.

### **Phân nhóm theo algebraic complexity**:

Tác giả chia 1,620 bề mặt thành ba nhóm dựa trên độ phức tạp đại số:

- **Nhóm phức tạp thấp (low complexity)**: 972 bề mặt.
- **Nhóm phức tạp trung bình (middle complexity)**: 486 bề mặt.
- **Nhóm phức tạp cao (high complexity)**: 162 bề mặt.
- **Tỷ lệ**: 6:3:1.

---

---

### Farthest Point Sampling (FPS)

### **1. Farthest Point Sampling (FPS) là gì?**

Farthest Point Sampling (FPS) là một thuật toán chọn **một tập hợp con** các điểm từ một tập hợp lớn hơn, sao cho các điểm được chọn phân bố đồng đều trong không gian. FPS đặc biệt phổ biến trong xử lý đám mây điểm (point cloud), nơi việc giảm số lượng điểm nhưng vẫn duy trì tính đại diện của hình học tổng thể là rất quan trọng.

FPS đảm bảo rằng khoảng cách nhỏ nhất giữa bất kỳ hai điểm nào trong tập con được chọn là lớn nhất có thể. Kết quả là tập con của các điểm sẽ bao phủ không gian ban đầu một cách hiệu quả, giảm thiểu mất mát thông tin hình học.

---

### **2. Các bước của thuật toán FPS**

FPS được thực hiện như sau:

1. **Khởi tạo**:
    - Chọn ngẫu nhiên một điểm $p_0$ từ tập điểm $P$ làm điểm đầu tiên trong tập con $S$.
    - Đặt $S = \{p_0\}$, nghĩa là $S$ ban đầu chỉ chứa điểm đầu tiên $p_0$.
2. **Lặp lại các bước sau**:
    - Với mỗi điểm $p \in P$, tính **khoảng cách nhỏ nhất** từ $p$ đến tất cả các điểm đã được chọn trong $S$:
        
        $$
        d(p) = \min_{q \in S} \|p - q\|
        $$
        
        $d(p)$ là khoảng cách từ $p$ đến điểm gần nhất trong $S$.
        
    - Chọn điểm $p_{\text{next}} \in P$ sao cho $d(p_{\text{next}})$ là lớn nhất:
        
        $$
        p_{\text{next}} = \arg\max_{p \in P} d(p)
        $$
        
    - Thêm $p_{\text{next}}$ vào $S = S \cup \{p_{\text{next}}\}$.
3. **Dừng thuật toán** khi tập $S$ đã chứa đủ số lượng điểm cần chọn.

**Kết quả**: Tập hợp $S$ gồm các điểm được chọn sao cho chúng cách nhau "xa nhất có thể", tạo ra sự phân bố đồng đều trong không gian.

---

### **3. Ý nghĩa của FPS**

- **Phân bố đều**: FPS đảm bảo rằng các điểm được chọn không bị tập trung quá nhiều ở một khu vực nhỏ mà phân bố đều khắp không gian.
- **Hiệu quả tính toán**: So với các thuật toán sampling khác (ví dụ k-means clustering), FPS có hiệu quả cao hơn trong việc giữ lại thông tin hình học của không gian.
- **Ứng dụng**: FPS thường được dùng trong mạng học sâu cho đám mây điểm, như PointNet hay PointNet++, để giảm số lượng điểm đầu vào trong khi vẫn giữ được cấu trúc hình học chính.

---

### **4. Ưu điểm và Hạn chế của FPS**

### **Ưu điểm**:

- **Đơn giản**: Thuật toán dễ hiểu và triển khai.
- **Đại diện tốt**: Điểm được chọn phân bố đều khắp không gian, giữ lại thông tin hình học tổng thể.
- **Hiệu quả**: Dùng tốt cho các bài toán giảm mẫu đầu vào (input sampling).

### **Hạn chế**:

- **Không tối ưu về tốc độ**: Với tập điểm lớn, việc tính toán khoảng cách ở mỗi bước có thể tốn thời gian.
- **Không hoàn toàn chính xác**: FPS không đảm bảo tối ưu toàn cục vì chỉ chọn dựa trên các bước lặp cục bộ.