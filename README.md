# Pointcloud_Transformation_and_Convert_2dmap

reference: https://github.com/koide3/pointcloud_to_2dmap
![transformed](https://github.com/user-attachments/assets/f12f98ed-f78c-423a-8948-58295e1a2e60)


## Requirement
- Eigen
- OpenCV
- PCL



```bash
# Usage
make
./pointcloud_to_2dmap <path to original pcd file> <option>
pointcloud_to_2dmap input_pcd dst_directory
```

```bash
# Help
pointcloud_to_2dmap:
  --help                                Produce help message
  -r [ --resolution ] arg (=0.1)        Pixel resolution (meters / pix)
  -w [ --map_width ] arg (=1024)        Map width [pix]
  -h [ --map_height ] arg (=1024)       Map height [pix]
  --min_points_in_pix arg (=2)          Min points in a occupied pix
  --max_points_in_pix arg (=5)          Max points in a pix for saturation
  --min_height arg (=0.5)               Min height of clipping range
  --max_height arg (=1)                 Max height of clipping range
  --input_pcd arg                       Input PCD file
  --dest_directory arg                  Destination directory
```

![Screenshot_20200716_160239](https://user-images.githubusercontent.com/31344317/87637926-e7adfc00-c77d-11ea-8987-19dffe614fa5.png)
