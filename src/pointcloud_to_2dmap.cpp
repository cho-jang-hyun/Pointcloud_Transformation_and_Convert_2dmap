#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <opencv2/opencv.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <pcl/common/transforms.h>
#include <Eigen/Dense>


class MapGenerater {
public:
  MapGenerater(boost::program_options::variables_map& vm)
  : resolution(vm["resolution"].as<double>()),
    m2pix(1.0 / resolution),
    map_width(vm["map_width"].as<int>()),
    map_height(vm["map_height"].as<int>()),
    min_points_in_pix(vm["min_points_in_pix"].as<int>()),
    max_points_in_pix(vm["max_points_in_pix"].as<int>()),
    min_height(vm["min_height"].as<double>()),
    max_height(vm["max_height"].as<double>())
  {}

  cv::Mat generate(const pcl::PointCloud<pcl::PointXYZ>& cloud) const {
    cv::Mat map(map_height, map_width, CV_32SC1, cv::Scalar::all(0));

    for(const auto& point: cloud) {
      if(point.z < min_height || point.z > max_height) {
        continue;
      }

      // 2D 맵에서 3D 맵의 시작점 설정
      // 현재는 (1500 x 1500) 이미지의 (1125, 1125) (3/4 지점)를 PCD 데이터의 (0,0,0)을 맞춘다
      int x = point.x * m2pix + map_width / 4 * 3;
      int y = -point.y * m2pix + map_width / 4 * 3;

      if(x < 0 || x >= map_width || y < 0 || y >= map_height) {
        continue;
      }

      map.at<int>(y, x) ++;
    }

    map -= min_points_in_pix;
    map.convertTo(map, CV_8UC1, - 255.0 / (max_points_in_pix - min_points_in_pix),  255);

    return map;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr transformPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float theta[], const Eigen::Vector3f& pivot) {
    // 회전 변환 행렬 생성
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();

    // 입력 받은 변환 각도
    float theta_x, theta_y, theta_z;

    theta_x = M_PI / 360 * theta[0];
    theta_y = M_PI / 360 * theta[1];
    theta_z = M_PI / 360 * theta[2];

    // 기준점으로 평행 이동
    transform.translation() = -pivot;
    pcl::PointCloud<pcl::PointXYZ>::Ptr shifted_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*cloud, *shifted_cloud, transform);

    // 회전 변환 적용
    transform.translation() = Eigen::Vector3f::Zero();
    transform.rotate(Eigen::AngleAxisf(theta_x, Eigen::Vector3f::UnitX()));
    transform.rotate(Eigen::AngleAxisf(theta_y, Eigen::Vector3f::UnitY()));
    transform.rotate(Eigen::AngleAxisf(theta_z, Eigen::Vector3f::UnitZ()));
    pcl::PointCloud<pcl::PointXYZ>::Ptr rotated_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*shifted_cloud, *rotated_cloud, transform);
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*rotated_cloud, *transformed_cloud, transform);

    return transformed_cloud;
}

public:
  const double resolution;    // meters per pixel
  const double m2pix;         // inverse resolution (pix/m)
  const int map_width;
  const int map_height;

  const int min_points_in_pix;
  const int max_points_in_pix;
  const double min_height;
  const double max_height;
};


int main(int argc, char** argv) {
  namespace po = boost::program_options;
  po::options_description description("pointcloud_to_2dmap");
  description.add_options()
    ("help", "Produce help message")
    ("resolution,r", po::value<double>()->default_value(0.1), "Pixel resolution (meters / pix)")
    ("map_width,w", po::value<int>()->default_value(1500), "Map width [pix]") // default: 1024
    ("map_height,h", po::value<int>()->default_value(1500), "Map height [pix]") // default: 1024
    ("min_points_in_pix", po::value<int>()->default_value(2), "Min points in a occupied pix")
    ("max_points_in_pix", po::value<int>()->default_value(5), "Max points in a pix for saturation")
    ("min_height", po::value<double>()->default_value(1), "Min height of clipping range") // default: 0.5
    ("max_height", po::value<double>()->default_value(2), "Max height of clipping range") // default: 1.0
    ("input_pcd", po::value<std::string>(), "Input PCD file")
    ("dest_directory", po::value<std::string>(), "Destination directory")
  ;

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv)
    .options(description)
    .positional(
      po::positional_options_description()
        .add("input_pcd", 1)
        .add("dest_directory", 1)
    ).run(), vm
  );
  po::notify(vm);

  if (vm.count("help")) {
      std::cout << description << std::endl;
      return 1;
  }

  std::cout << "input_pcd     :" << vm["input_pcd"].as<std::string>() << std::endl;
  std::cout << "dest_directory:" << vm["dest_directory"].as<std::string>() << std::endl;
  std::cout << "resolution    :" << vm["resolution"].as<double>() << std::endl;
  auto cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  if(pcl::io::loadPCDFile(vm["input_pcd"].as<std::string>(), *cloud)) {
    std::cerr << "failed to open the input cloud" << std::endl;
    return 1;
  }

  // MapGenerater 인스턴스 생성
  MapGenerater transformation_map_generater(vm);
  MapGenerater map2d_generater(vm);
  cv::Mat map;

  // 모드 설정
  bool transformation_mode = true; // true: pcd 변환 및 별도 저장과 2d map 저장, false: 2d map 저장

  /*
  아래 if 문은 pcd 파일을 translation 하고 다른 파일로 저장하는 부분
  */
  if (transformation_mode){

    // rotation(회전) 각도 설정: (x,y,z 순으로 degree 단위)
    float theta[3] = {5, -20, 15.5}; // 임의로 찾은 최적의 값 (5, -20, 15.75도)

    // translation(평행 이동) 설정: x는 위아래, y는 좌우
    // 맵을 만드려고 하는 pointcloud의 가장 왼쪽 아래 영역으로 
    Eigen::Vector3f pivot(2.6, -7.2, 0);

    // 변환 함수
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud = transformation_map_generater.transformPointCloud(cloud, theta, pivot);

    // transform 결과를 별도의 pcd 저장, 필요하지 않은 경우 주석 처리
    std::string output_filename = "transformed2_9F.pcd"; // 출력 PCD 파일 경로
    std::cout << "PointCloud transform start!" << std::endl;
    pcl::io::savePCDFileASCII(output_filename, *transformed_cloud); // 별도의 PCD 파일 저장
    std::cout << "transformed PointCloud saved!" << std::endl;

    //cv::Mat 
    map = map2d_generater.generate(*transformed_cloud);
  }

  else{
    //cv::Mat 
    map = map2d_generater.generate(*cloud);
  }

  std::string destination = vm["dest_directory"].as<std::string>();
  if(!boost::filesystem::exists(destination)) {
    boost::filesystem::create_directories(destination);
  }

  // 파일 이름 지정
  cv::imwrite(destination + "/sliced_pretransformed_map.png", map);
  
  std::ofstream ofs(destination + "/map.yaml");
  ofs << "image: map.png" << std::endl;
  ofs << "resolution: " << map2d_generater.resolution << std::endl;
  ofs << "origin: [" << -map2d_generater.resolution * map2d_generater.map_width / 2 << ", " << -map2d_generater.resolution * map2d_generater.map_height / 2 << ", 0.0]" << std::endl;
  ofs << "occupied_thresh: 0.5" << std::endl;
  ofs << "free_thresh: 0.2" << std::endl;
  ofs << "negate: 0" << std::endl;
  std::cout << "map image saved!" << std::endl;

  return 0;
}