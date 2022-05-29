# rrt_practice

## RRTとは
1999年に提案された"Rapidly exploring random tree"と呼ばれる経路計画法の一種．
(直訳すると，高速探索ランダムツリー)

状態空間(2次元:x-y空間，3次元:x-y-z空間)にてランダムサンプリングを実施し，そのサンプリング点に一番近いノードをある一定距離づつ伸ばしていくことで経路を探索する．

![image](https://user-images.githubusercontent.com/60972444/170849936-1084cecf-de10-40f3-83b3-1964e906727e.png)


### 抽象的イメージ
ランダムに道(点)を探索し，目的地まで上手く行けたらイイね

### 使用用途
- 走行ロボットの軌道計画
- ロボットアームの軌道生成

### 特徴
- **利点**
    - 漸近的に探索が完了する
    - 状態空間が高次元でも探索可能
    - 原理が単純明快で実装が容易
    - 制約条件を簡単に追加できる
- **欠点**
    - 経路の最適性が担保されない
    - 比較的うねった軌道を作成しがち
    - オフライン計算を使って軌道生成を高速化するのが難

---
## アルゴリズム
1. 状態空間上にて，初期位置ノードを決める
2. 状態空間上のある特定のエリアにて，ランダムサンプリングを施す
3. ランダムサンプリングされた点に最も近いノードを選択
4. 事前に決めた量だけ，その方向に伸ばす
5. 目的(ゴール)まで繰り返す

---
## 実装
`rrt.py`: 目的(ゴール)や障害物を特に定めず，純粋にRRTの経路生成の流れを可視化する
