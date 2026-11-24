# -*- coding: utf-8 -*-
"""
@new 2025-12-22: 空间索引模块
========================

实现 R-tree 空间索引用于加速土地匹配过程中的邻近查询

理论依据：
1. Guttman (1984): R-trees: A dynamic index structure for spatial searching
2. 在大规模模拟中，O(N²) 的邻近查询是性能瓶颈
3. R-tree 可将查询复杂度降至 O(log N)

应用场景：
- 土地流转匹配时查找邻近地块
- 农业服务市场空间匹配
- 社会网络形成时的邻域计算

使用方法：
    from modules.spatial_index import SpatialIndex
    
    # 创建索引
    index = SpatialIndex()
    
    # 批量添加对象
    for patch in patches:
        index.insert(patch.unique_id, patch.coordinates)
    
    # 查询邻近对象
    nearby_ids = index.query_neighbors(center_coords, radius=5.0)
"""

import numpy as np
import hashlib
from typing import List, Tuple, Any, Dict, Set, Optional
from collections import defaultdict


def _stable_int_id(value: Any, modulus: int = 2**31) -> int:
    """Create a process-stable integer id instead of relying on Python's salted hash()."""
    digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulus


def _stable_patch_id(patch: Any) -> str:
    """Return a process-stable patch identifier for spatial indexing."""
    getter = getattr(patch, "get_patch_id", None)
    if callable(getter):
        return str(getter())

    patch_id = getattr(patch, "patch_id", None)
    if patch_id is not None:
        return str(patch_id)

    coords = getattr(patch, "coordinates", None) or (0, 0)
    owner = getattr(patch, "owner", None)
    owner_id = getattr(owner, "unique_id", "unassigned")
    area = float(getattr(patch, "area", 0.0) or 0.0)
    soil_quality = float(getattr(patch, "soil_quality", 0.0) or 0.0)
    climate_zone = getattr(patch, "climate_zone", "unknown")
    payload = f"{owner_id}|{climate_zone}|{coords[0]}|{coords[1]}|{area:.8f}|{soil_quality:.8f}"
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]
    return f"patch_{digest}"


class SpatialIndex:
    """
    空间索引类
    
    使用简化的网格索引实现（Grid Index），在不依赖 rtree 库的情况下
    提供高效的空间查询。对于中小规模模拟（<10000个对象），
    网格索引性能接近 R-tree。
    
    如果安装了 rtree 库，会自动使用 R-tree 实现以获得最佳性能。
    """
    
    def __init__(self, cell_size: float = 5.0, use_rtree: bool = True):
        """
        初始化空间索引
        
        Args:
            cell_size: 网格单元大小（用于网格索引）
            use_rtree: 是否尝试使用 rtree 库
        """
        self.cell_size = cell_size
        self.use_rtree = use_rtree
        self._rtree_index = None
        self._grid_index = defaultdict(set)  # {(grid_x, grid_y): set of obj_ids}
        self._objects = {}  # {obj_id: coordinates}
        
        # 尝试使用 rtree
        if use_rtree:
            try:
                from rtree import index
                p = index.Property()
                p.dimension = 2
                self._rtree_index = index.Index(properties=p)
                self._using_rtree = True
            except ImportError:
                self._using_rtree = False
                # print("[INFO] rtree 库未安装，使用网格索引替代")
        else:
            self._using_rtree = False
    
    def _get_grid_cell(self, coords: Tuple[float, float]) -> Tuple[int, int]:
        """计算坐标所在的网格单元"""
        return (int(coords[0] // self.cell_size), 
                int(coords[1] // self.cell_size))
    
    def insert(self, obj_id: Any, coordinates: Tuple[float, float]):
        """
        插入对象到索引
        
        Args:
            obj_id: 对象唯一标识
            coordinates: (x, y) 坐标
        """
        if obj_id in self._objects:
            # 如果已存在，先删除旧的
            self.remove(obj_id)
        
        self._objects[obj_id] = coordinates
        
        if self._using_rtree:
            # R-tree 索引：使用点作为边界框
            x, y = coordinates
            self._rtree_index.insert(_stable_int_id(obj_id), (x, y, x, y), obj=obj_id)
        else:
            # 网格索引
            cell = self._get_grid_cell(coordinates)
            self._grid_index[cell].add(obj_id)
    
    def remove(self, obj_id: Any):
        """
        从索引中删除对象
        
        Args:
            obj_id: 对象唯一标识
        """
        if obj_id not in self._objects:
            return
        
        coordinates = self._objects[obj_id]
        
        if self._using_rtree:
            x, y = coordinates
            self._rtree_index.delete(_stable_int_id(obj_id), (x, y, x, y))
        else:
            cell = self._get_grid_cell(coordinates)
            self._grid_index[cell].discard(obj_id)
        
        del self._objects[obj_id]
    
    def update(self, obj_id: Any, new_coordinates: Tuple[float, float]):
        """
        更新对象位置
        
        Args:
            obj_id: 对象唯一标识
            new_coordinates: 新的 (x, y) 坐标
        """
        self.remove(obj_id)
        self.insert(obj_id, new_coordinates)
    
    def query_neighbors(self, center: Tuple[float, float], radius: float) -> List[Any]:
        """
        查询指定半径内的所有对象
        
        Args:
            center: 查询中心点 (x, y)
            radius: 搜索半径
        
        Returns:
            在半径范围内的对象 ID 列表
        """
        if self._using_rtree:
            return self._query_rtree(center, radius)
        else:
            return self._query_grid(center, radius)
    
    def _query_rtree(self, center: Tuple[float, float], radius: float) -> List[Any]:
        """使用 R-tree 查询"""
        x, y = center
        # 创建搜索边界框
        bbox = (x - radius, y - radius, x + radius, y + radius)
        
        # 获取边界框内的候选对象
        candidates = list(self._rtree_index.intersection(bbox, objects=True))
        
        # 精确过滤：检查欧氏距离
        results = []
        radius_sq = radius * radius
        for item in sorted(candidates, key=lambda candidate: str(candidate.object)):
            obj_id = item.object
            if obj_id in self._objects:
                coords = self._objects[obj_id]
                dist_sq = (coords[0] - x) ** 2 + (coords[1] - y) ** 2
                if dist_sq <= radius_sq:
                    results.append(obj_id)
        
        return sorted(results, key=str)
    
    def _query_grid(self, center: Tuple[float, float], radius: float) -> List[Any]:
        """使用网格索引查询"""
        x, y = center
        
        # 计算需要检查的网格单元范围
        cells_to_check = int(np.ceil(radius / self.cell_size)) + 1
        center_cell = self._get_grid_cell(center)
        
        candidates = set()
        for dx in range(-cells_to_check, cells_to_check + 1):
            for dy in range(-cells_to_check, cells_to_check + 1):
                cell = (center_cell[0] + dx, center_cell[1] + dy)
                if cell in self._grid_index:
                    candidates.update(self._grid_index[cell])
        
        # 精确过滤
        results = []
        radius_sq = radius * radius
        for obj_id in sorted(candidates, key=str):
            coords = self._objects[obj_id]
            dist_sq = (coords[0] - x) ** 2 + (coords[1] - y) ** 2
            if dist_sq <= radius_sq:
                results.append(obj_id)
        
        return sorted(results, key=str)
    
    def query_k_nearest(self, center: Tuple[float, float], k: int = 5, 
                        max_radius: float = 50.0) -> List[Tuple[Any, float]]:
        """
        查询 k 个最近邻
        
        Args:
            center: 查询中心点 (x, y)
            k: 返回的最近邻数量
            max_radius: 最大搜索半径
        
        Returns:
            (obj_id, distance) 元组列表，按距离排序
        """
        x, y = center
        
        # 使用递增半径策略
        radius = self.cell_size
        results = []
        
        while radius <= max_radius:
            candidates = self.query_neighbors(center, radius)
            
            # 计算距离并排序
            distances = []
            for obj_id in candidates:
                coords = self._objects[obj_id]
                dist = np.sqrt((coords[0] - x) ** 2 + (coords[1] - y) ** 2)
                distances.append((obj_id, dist))
            
            distances.sort(key=lambda x: (x[1], str(x[0])))
            
            # 检查是否找到足够的邻居
            if len(distances) >= k:
                return distances[:k]
            
            # 扩大搜索半径
            radius *= 2
        
        # 返回找到的所有邻居（可能少于 k 个）
        return sorted(
            [
                (
                    obj_id,
                    np.sqrt(
                        (self._objects[obj_id][0] - x) ** 2 +
                        (self._objects[obj_id][1] - y) ** 2
                    ),
                )
                for obj_id in self._objects
            ],
            key=lambda x: (x[1], str(x[0])),
        )[:k]
    
    def get_all_ids(self) -> List[Any]:
        """获取所有索引中的对象 ID"""
        return sorted(self._objects.keys(), key=str)
    
    def get_coordinates(self, obj_id: Any) -> Optional[Tuple[float, float]]:
        """获取对象坐标"""
        return self._objects.get(obj_id)
    
    def clear(self):
        """清空索引"""
        self._objects.clear()
        self._grid_index.clear()
        
        if self._using_rtree:
            from rtree import index
            p = index.Property()
            p.dimension = 2
            self._rtree_index = index.Index(properties=p)
    
    def __len__(self):
        return len(self._objects)
    
    def __contains__(self, obj_id):
        return obj_id in self._objects


def area_moore_neighbors(actors, seed=17):
    """Synthetic grid of the actual physical area ledgers, not surveyed coordinates."""
    from collections import defaultdict
    from scipy.sparse import csr_matrix
    import numpy as np
    regions=defaultdict(list)
    for actor in actors:
        regions[actor.zone].append(actor)
    coords={}; anchor={}; areas={}; rights={}; zone_by_plot={}
    for region,group in sorted(regions.items()):
        farms=sorted((a for a in group if a.area_kind=='household'),key=lambda a:str(a.unique_id))
        if not farms:
            continue
        rng=np.random.default_rng([seed,int(region)])
        side=int(np.ceil(np.sqrt(len(farms))))
        for k,a in zip(rng.permutation(len(farms)),farms):
            coords[a.unique_id]=(int(region),int(k%side),int(k//side))
            areas[a.unique_id]=a.base_area
            rights[a.unique_id]=a.lease_right
            zone_by_plot[a.unique_id]=int(region)
        for a in sorted((a for a in group if a.area_kind!='household'),key=lambda a:str(a.unique_id)):
            from hashlib import sha256
            h=int.from_bytes(sha256(f'{seed}:{region}:{a.unique_id}'.encode()).digest()[:8],'big')
            anchor[a.unique_id]=farms[h%len(farms)].unique_id
    ids=list(coords)
    idx={k:i for i,k in enumerate(ids)}
    by_coord={v:k for k,v in coords.items()}
    pairs=[(i,idx[by_coord[z,x+dx,y+dy]]) for i,k in enumerate(ids)
           for z,x,y in [coords[k]] for dx in (-1,0,1) for dy in (-1,0,1)
           if (dx or dy) and (z,x+dx,y+dy) in by_coord]
    ii,jj=zip(*pairs) if pairs else ([],[])
    w=csr_matrix((np.ones(len(pairs),dtype=bool),(ii,jj)),shape=(len(ids),len(ids)))
    assert not (w!=w.T).nnz and np.diff(w.indptr).max(initial=0)<=8
    return dict(ids=ids,index=idx,coordinates=coords,area=areas,rights=rights,zone=zone_by_plot,
                anchors={k:k for k in ids}|anchor,neighbors=w,synthetic=True)


class PatchSpatialIndex:
    """
    土地地块专用空间索引
    
    封装了 SpatialIndex，专门用于土地流转中的邻近查询
    """
    
    def __init__(self, patches: List = None, cell_size: float = 5.0):
        """
        初始化地块空间索引
        
        Args:
            patches: 初始地块列表
            cell_size: 网格单元大小
        """
        self._index = SpatialIndex(cell_size=cell_size)
        self._patches = {}  # {patch_id: patch_object}
        
        if patches:
            self.build_index(patches)
    
    def build_index(self, patches: List):
        """
        从地块列表构建索引
        
        Args:
            patches: 地块对象列表，需要有 coordinates 属性
        """
        self._index.clear()
        self._patches.clear()
        
        for patch in patches:
            if hasattr(patch, 'coordinates') and patch.coordinates:
                patch_id = _stable_patch_id(patch)
                self._index.insert(patch_id, patch.coordinates)
                self._patches[patch_id] = patch
    
    def add_patch(self, patch):
        """添加单个地块到索引"""
        if hasattr(patch, 'coordinates') and patch.coordinates:
            patch_id = _stable_patch_id(patch)
            self._index.insert(patch_id, patch.coordinates)
            self._patches[patch_id] = patch
    
    def remove_patch(self, patch):
        """从索引中移除地块"""
        patch_id = _stable_patch_id(patch)
        if patch_id in self._patches:
            self._index.remove(patch_id)
            del self._patches[patch_id]
    
    def find_neighbors(self, patch, radius: float = 5.0) -> List:
        """
        查找给定地块的邻近地块
        
        Args:
            patch: 目标地块
            radius: 搜索半径
        
        Returns:
            邻近地块列表
        """
        if not hasattr(patch, 'coordinates') or not patch.coordinates:
            return []
        
        neighbor_ids = self._index.query_neighbors(patch.coordinates, radius)
        patch_id = _stable_patch_id(patch)
        
        # 排除自身
        neighbors = [
            self._patches[nid] for nid in neighbor_ids
            if nid != patch_id and nid in self._patches
        ]
        return sorted(neighbors, key=_stable_patch_id)
    
    def find_contiguous(self, patches: List, max_gap: float = 1.5) -> List[Set]:
        """
        查找连片地块组
        
        使用连通分量算法识别空间上连续的地块群组
        
        Args:
            patches: 待分析的地块列表
            max_gap: 判定为连片的最大间距
        
        Returns:
            连片地块组列表，每组为一个 set
        """
        if not patches:
            return []
        
        # 构建临时索引
        patch_ids = {_stable_patch_id(p): p for p in patches if hasattr(p, 'coordinates')}
        
        # 使用并查集（Union-Find）
        parent = {pid: pid for pid in patch_ids}
        rank = {pid: 0 for pid in patch_ids}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
        
        # 检查所有地块对
        patch_list = sorted(patch_ids.keys(), key=str)
        for i, pid1 in enumerate(patch_list):
            p1 = patch_ids[pid1]
            # 查询邻近
            neighbors = self._index.query_neighbors(p1.coordinates, max_gap)
            for pid2 in neighbors:
                if pid2 in patch_ids and pid2 != pid1:
                    union(pid1, pid2)
        
        # 收集连通分量
        groups = defaultdict(set)
        for pid in patch_ids:
            root = find(pid)
            groups[root].add(patch_ids[pid])
        
        contiguous_groups = [group for group in groups.values() if len(group) > 1]
        return sorted(
            contiguous_groups,
            key=lambda group: min(_stable_patch_id(patch) for patch in group),
        )
    
    def get_patch_count(self) -> int:
        """获取索引中的地块数量"""
        return len(self._patches)


# 全局索引实例（用于模型级别的空间查询）
_global_patch_index: Optional[PatchSpatialIndex] = None


def get_global_patch_index() -> Optional[PatchSpatialIndex]:
    """获取全局地块索引"""
    return _global_patch_index


def set_global_patch_index(index: PatchSpatialIndex):
    """设置全局地块索引"""
    global _global_patch_index
    _global_patch_index = index


def build_global_patch_index(patches: List, cell_size: float = 5.0) -> PatchSpatialIndex:
    """
    构建并设置全局地块索引
    
    Args:
        patches: 地块列表
        cell_size: 网格单元大小
    
    Returns:
        构建的索引对象
    """
    global _global_patch_index
    _global_patch_index = PatchSpatialIndex(patches, cell_size)
    return _global_patch_index


if __name__ == "__main__":
    # 测试代码
    print("测试空间索引模块...")
    
    # 创建测试数据
    rng = build_numpy_rng(42, "spatial_index_demo")
    n_points = 1000
    coords = rng.random_sample((n_points, 2)) * 100
    
    # 测试基本功能
    index = SpatialIndex(cell_size=10.0)
    
    for i in range(n_points):
        index.insert(i, (coords[i, 0], coords[i, 1]))
    
    # 测试邻近查询
    center = (50.0, 50.0)
    radius = 10.0
    neighbors = index.query_neighbors(center, radius)
    print(f"在 ({center}) 半径 {radius} 内找到 {len(neighbors)} 个邻居")
    
    # 验证结果
    correct_count = sum(1 for i in range(n_points) 
                        if np.sqrt((coords[i, 0] - 50)**2 + (coords[i, 1] - 50)**2) <= radius)
    print(f"暴力搜索验证: {correct_count} 个邻居")
    assert len(neighbors) == correct_count, "邻近查询结果不正确!"
    
    # 测试 k 最近邻
    k_nearest = index.query_k_nearest(center, k=10)
    print(f"10 个最近邻:")
    for obj_id, dist in k_nearest[:5]:
        print(f"  ID={obj_id}, 距离={dist:.2f}")
    
    print("\n空间索引模块测试通过!")
from core.reproducibility import build_numpy_rng
