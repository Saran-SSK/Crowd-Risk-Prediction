import numpy as np


class RegionAnalyzer:
    """
    Analyzes crowd density map by dividing it into spatial regions.
    
    This class takes a 2D CSRNet density map and divides it into a grid of regions
    (e.g., 3x3). For each region, it calculates statistics such as the crowd count
    by summing the density values within that region.
    
    This enables region-wise crowd analysis without modifying the underlying CSRNet model.
    """

    def __init__(self, rows=3, cols=3):
        """
        Initialize the RegionAnalyzer with a specified grid size.
        
        Args:
            rows (int): Number of rows in the grid. Default is 3.
            cols (int): Number of columns in the grid. Default is 3.
        """
        self.rows = rows
        self.cols = cols

    def _get_region_boundaries(self, height, width):
        """
        Calculate the boundaries for each region in the grid.
        
        This helper method ensures that analyze_density() and analyze_motion()
        use the exact same region boundaries for consistency.
        
        Args:
            height (int): Height of the input array (density map or flow magnitude)
            width (int): Width of the input array
        
        Returns:
            list: A list of tuples containing (row_start, row_end, col_start, col_end)
                  for each region in row-major order.
        """
        # Calculate the height and width of each region
        region_height = height // self.rows
        region_width = width // self.cols
        
        boundaries = []
        
        # Iterate over each region in the grid
        for row in range(self.rows):
            for col in range(self.cols):
                # Calculate the boundaries of the current region
                row_start = row * region_height
                row_end = (row + 1) * region_height if row < self.rows - 1 else height
                col_start = col * region_width
                col_end = (col + 1) * region_width if col < self.cols - 1 else width
                
                boundaries.append((row_start, row_end, col_start, col_end))
        
        return boundaries

    def analyze_density(self, density_map):
        """
        Analyze the CSRNet density map by dividing it into regions.
        
        This method takes a 2D density map (output from CSRNet) and divides it
        into a grid of regions. For each region, it calculates the crowd count
        by summing the density values within that region's boundaries.
        
        Args:
            density_map (np.ndarray): 2D numpy array representing the crowd density map.
                                      Shape should be (H, W) where H is height and W is width.
        
        Returns:
            list: A list of dictionaries, where each dictionary contains statistics
                  for one region. Each dictionary has the following keys:
                  - region_id (str): Unique identifier for the region (e.g., "R0_C0")
                  - row (int): Row index of the region (0 to rows-1)
                  - column (int): Column index of the region (0 to cols-1)
                  - density_count (float): Sum of density values in this region
                  - bbox (tuple): Bounding box as (row_start, row_end, col_start, col_end)
        
        Example:
            >>> analyzer = RegionAnalyzer(rows=3, cols=3)
            >>> density_map = np.random.rand(100, 100)  # Example density map
            >>> regions = analyzer.analyze_density(density_map)
            >>> print(len(regions))  # Should be 9 for 3x3 grid
        """
        # Get the dimensions of the density map
        height, width = density_map.shape
        
        # Get region boundaries using the helper method
        boundaries = self._get_region_boundaries(height, width)
        
        regions = []
        region_idx = 0
        
        # Iterate over each region in the grid
        for row in range(self.rows):
            for col in range(self.cols):
                # Get the boundaries for this region
                row_start, row_end, col_start, col_end = boundaries[region_idx]
                
                # Extract the density values for this region
                region_density = density_map[row_start:row_end, col_start:col_end]
                
                # Calculate the crowd count for this region (sum of density values)
                density_count = region_density.sum()
                
                # Create a region identifier
                region_id = f"R{row}_C{col}"
                
                # Store the region statistics
                region_info = {
                    "region_id": region_id,
                    "row": row,
                    "column": col,
                    "density_count": density_count,
                    "bbox": (row_start, row_end, col_start, col_end)
                }
                
                regions.append(region_info)
                region_idx += 1
        
        return regions

    def analyze_motion(self, flow_magnitude):
        """
        Analyze the optical flow magnitude by dividing it into regions.
        
        This method takes a 2D optical flow magnitude array and divides it
        into the same grid of regions used by analyze_density(). For each region,
        it calculates the average motion magnitude within that region's boundaries.
        
        This ensures that density and motion analysis correspond to the exact same
        spatial regions for consistent region-wise risk assessment.
        
        Args:
            flow_magnitude (np.ndarray): 2D numpy array representing the optical flow
                                         magnitude. Shape should be (H, W) where H is
                                         height and W is width.
        
        Returns:
            list: A list of dictionaries, where each dictionary contains statistics
                  for one region. Each dictionary has the following keys:
                  - region_id (str): Unique identifier for the region (e.g., "R0_C0")
                  - row (int): Row index of the region (0 to rows-1)
                  - column (int): Column index of the region (0 to cols-1)
                  - average_motion (float): Average optical flow magnitude in this region
                  - bbox (tuple): Bounding box as (row_start, row_end, col_start, col_end)
        
        Example:
            >>> analyzer = RegionAnalyzer(rows=3, cols=3)
            >>> flow_magnitude = np.random.rand(100, 100)  # Example flow magnitude
            >>> regions = analyzer.analyze_motion(flow_magnitude)
            >>> print(len(regions))  # Should be 9 for 3x3 grid
        """
        # Get the dimensions of the flow magnitude array
        height, width = flow_magnitude.shape
        
        # Get region boundaries using the helper method (same as density)
        boundaries = self._get_region_boundaries(height, width)
        
        regions = []
        region_idx = 0
        
        # Iterate over each region in the grid
        for row in range(self.rows):
            for col in range(self.cols):
                # Get the boundaries for this region
                row_start, row_end, col_start, col_end = boundaries[region_idx]
                
                # Extract the flow magnitude values for this region
                region_flow = flow_magnitude[row_start:row_end, col_start:col_end]
                
                # Calculate the average motion for this region
                average_motion = region_flow.mean()
                
                # Create a region identifier
                region_id = f"R{row}_C{col}"
                
                # Store the region statistics
                region_info = {
                    "region_id": region_id,
                    "row": row,
                    "column": col,
                    "average_motion": average_motion,
                    "bbox": (row_start, row_end, col_start, col_end)
                }
                
                regions.append(region_info)
                region_idx += 1
        
        return regions

    def analyze_density_change(self, previous_regions, current_regions):
        """
        Analyze the change in regional density between two consecutive frames.
        
        This method takes the regional density results from two consecutive frames
        and calculates the change in density for each region. This enables temporal
        trend analysis to identify regions where crowd density is increasing or decreasing.
        
        Args:
            previous_regions (list): List of region dictionaries from analyze_density() 
                                    for the previous frame.
            current_regions (list): List of region dictionaries from analyze_density()
                                   for the current frame.
        
        Returns:
            list: A list of dictionaries, where each dictionary contains change statistics
                  for one region. Each dictionary has the following keys:
                  - region_id (str): Unique identifier for the region (e.g., "R0_C0")
                  - row (int): Row index of the region (0 to rows-1)
                  - column (int): Column index of the region (0 to cols-1)
                  - previous_density (float): Density count in the previous frame
                  - current_density (float): Density count in the current frame
                  - density_change (float): Change in density (current - previous)
                  - density_change_ratio (float): Ratio of change relative to previous density
                  - bbox (tuple): Bounding box as (row_start, row_end, col_start, col_end)
        
        Example:
            >>> analyzer = RegionAnalyzer(rows=3, cols=3)
            >>> prev_regions = analyzer.analyze_density(prev_density_map)
            >>> curr_regions = analyzer.analyze_density(curr_density_map)
            >>> changes = analyzer.analyze_density_change(prev_regions, curr_regions)
            >>> print(len(changes))  # Should be 9 for 3x3 grid
        """
        # Ensure both lists have the same number of regions
        if len(previous_regions) != len(current_regions):
            raise ValueError(
                f"Previous regions count ({len(previous_regions)}) does not match "
                f"current regions count ({len(current_regions)})"
            )
        
        changes = []
        
        # Iterate over corresponding regions
        for prev_region, curr_region in zip(previous_regions, current_regions):
            # Verify region IDs match
            if prev_region['region_id'] != curr_region['region_id']:
                raise ValueError(
                    f"Region ID mismatch: {prev_region['region_id']} vs {curr_region['region_id']}"
                )
            
            previous_density = prev_region['density_count']
            current_density = curr_region['density_count']
            
            # Calculate density change
            density_change = current_density - previous_density
            
            # Calculate density change ratio safely (prevent division by zero and Inf)
            # A safe epsilon prevents numerical instability if previous density is near zero
            eps_density = 1e-6
            if previous_density > eps_density:
                density_change_ratio = float(density_change / previous_density)
            else:
                density_change_ratio = 0.0
            
            # Store the change statistics
            change_info = {
                "region_id": prev_region['region_id'],
                "row": prev_region['row'],
                "column": prev_region['column'],
                "previous_density": previous_density,
                "current_density": current_density,
                "density_change": density_change,
                "density_change_ratio": density_change_ratio,
                "bbox": prev_region['bbox']
            }
            
            changes.append(change_info)
        
        return changes

    def analyze_convergence(self, flow_x, flow_y):
        """
        Analyze crowd convergence by measuring magnitude-weighted net inward optical flow.
        
        MATHEMATICAL DEFINITION:
        For each pixel i in the region:
          - u_i is the unit vector pointing from pixel (y_i, x_i) toward the region center (c_y, c_x).
          - v_i = (flow_x_i, flow_y_i) is the optical flow vector with magnitude |v_i|.
          - projection_i = dot(v_i, u_i) is the flow component directed toward the center.
        
        The magnitude-weighted net inward flux across all N pixels in the region is:
          net_inward = sum(projection_i) / (sum(|v_i|) + epsilon)
        
        The convergence score C is defined as:
          C = clip(max(0.0, net_inward), 0.0, 1.0)
        
        ENGINEERING PROPERTIES:
        - Zero flow: sum(|v_i|) < epsilon -> C = 0.0
        - Pure inward flow (gathering): C -> 1.0
        - Pure outward flow (dispersal): net_inward < 0 -> C = 0.0
        - Uniform directional flow: opposite halves cancel out -> C ≈ 0.0
        - Random / mixed flow: expected to have low net inward convergence -> C ≈ 0.0
        
        This formulation avoids the mathematical artifact of positive-only cosine averaging,
        which previously produced an artificial ~0.637 baseline across all regions.
        
        Args:
            flow_x (np.ndarray): Horizontal optical flow component. Shape (H, W).
            flow_y (np.ndarray): Vertical optical flow component. Shape (H, W).
        
        Returns:
            list: Dictionaries containing convergence statistics for each region.
        """
        # Get the dimensions of the flow arrays
        height, width = flow_x.shape
        
        # Get region boundaries using the helper method (same as density and motion)
        boundaries = self._get_region_boundaries(height, width)
        
        regions = []
        region_idx = 0
        eps_flow = 1e-7
        
        # Iterate over each region in the grid
        for row in range(self.rows):
            for col in range(self.cols):
                # Get the boundaries for this region
                row_start, row_end, col_start, col_end = boundaries[region_idx]
                
                # Extract the flow vectors for this region
                region_flow_x = flow_x[row_start:row_end, col_start:col_end]
                region_flow_y = flow_y[row_start:row_end, col_start:col_end]
                
                # Calculate average flow components for this region
                average_flow_x = float(region_flow_x.mean())
                average_flow_y = float(region_flow_y.mean())
                
                # Calculate the center of the region
                region_center_y = (row_start + row_end) / 2.0
                region_center_x = (col_start + col_end) / 2.0
                
                # Create coordinate grids
                y_coords, x_coords = np.meshgrid(
                    np.arange(row_start, row_end),
                    np.arange(col_start, col_end),
                    indexing='ij'
                )
                
                # Calculate inward direction vectors (from each pixel toward region center)
                inward_y = region_center_y - y_coords
                inward_x = region_center_x - x_coords
                
                # Normalize inward direction vectors
                inward_dist = np.sqrt(inward_x**2 + inward_y**2)
                safe_dist = np.where(inward_dist == 0, 1.0, inward_dist)
                u_x = np.where(inward_dist == 0, 0.0, inward_x / safe_dist)
                u_y = np.where(inward_dist == 0, 0.0, inward_y / safe_dist)
                
                # Optical flow magnitude at each pixel
                flow_magnitude = np.sqrt(region_flow_x**2 + region_flow_y**2)
                sum_magnitude = float(np.sum(flow_magnitude))
                
                # Safe handling for zero or near-zero flow
                if sum_magnitude < eps_flow:
                    convergence_score = 0.0
                else:
                    # Inward projection = dot(v, u) = v_x * u_x + v_y * u_y
                    inward_projection = region_flow_x * u_x + region_flow_y * u_y
                    sum_projection = float(np.sum(inward_projection))
                    
                    # Magnitude-weighted net inward convergence
                    net_inward = sum_projection / (sum_magnitude + eps_flow)
                    
                    # Positive inward motion contributes to risk; outward motion (net < 0) yields 0
                    convergence_score = float(np.clip(max(0.0, net_inward), 0.0, 1.0))
                
                # Create a region identifier
                region_id = f"R{row}_C{col}"
                
                # Store the region statistics
                region_info = {
                    "region_id": region_id,
                    "row": row,
                    "column": col,
                    "average_flow_x": average_flow_x,
                    "average_flow_y": average_flow_y,
                    "convergence_score": convergence_score,
                    "bbox": (row_start, row_end, col_start, col_end)
                }
                
                regions.append(region_info)
                region_idx += 1
        
        return regions
