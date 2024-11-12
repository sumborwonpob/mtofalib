#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <stdio.h>

#include "rclcpp/rclcpp.hpp"
#include "mtofalib/msg/mtof.hpp"

#include "VL53L5CX_I2C_DRIVER/vl53l5cx_api.h"
#include "VL53L5CX_I2C_DRIVER/vl53l5cx_api.c"
#include "VL53L5CX_I2C_DRIVER/platform.h"
#include "VL53L5CX_I2C_DRIVER/platform.c"

#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;
// ::::::::::::::::::::: Parameters ::::::::::::::::::::::::
char * DEVICE_PATH = "/dev/i2c-5";
auto MTOF_PIXEL_MODE = VL53L5CX_RESOLUTION_8X8; // Resolution VL53L5CX_RESOLUTION_8X8 or VL53L5CX_RESOLUTION_4X4
int MTOF_FREQ = 30; // 8x8 Max 15 Hz, 4x4 Max 30 Hz

class MtofPublisher : public rclcpp::Node
{
public:
	rclcpp::Publisher<mtofalib::msg::Mtof>::SharedPtr mtof_pub;
	MtofPublisher() : Node("mmtof_publisher") 
	{
		mtof_pub = this->create_publisher<mtofalib::msg::Mtof>("/mtof/data", 1);

        // Instantiate variables
		uint8_t 				status, loop, isAlive, isReady, i;
		VL53L5CX_Configuration 	Dev;			
		VL53L5CX_ResultsData 	Results;		

        // Create platform variable and adjust address
		Dev.platform.address = 0x52; // Put your sensor address, default is 0x52
		auto p_dev = &Dev.platform;
		
        // Initialize I2C communication with the sensor
		int init_ret = VL53L5CX_Comms_Init(p_dev, DEVICE_PATH);
		if(init_ret != 0)
        {
            exit(init_ret);
            cout << "Sensor initialization failed!" << endl;
        } 

        // Check sensor status
		printf("Checking I2C device...\n");
		status = vl53l5cx_is_alive(&Dev, &isAlive);
		if(!isAlive || status)
		{
			printf(" VL53L5CX not detected at requested address\n");
			exit(-1);
		}
		else
		{
			printf(" VL53L5CX is there!\n");
		}

        // Load firmware
		printf("Loading firmware...\n");
		status = vl53l5cx_init(&Dev);
		if(status)
		{
			printf(" VL53L5CX ULD Loading failed\n");
			exit(-1);
		}
		else
		{
			printf(" Firmware loading completed!\n");
		}

		printf("VL53L5CX ULD ready ! (Version : %s)\n",
				VL53L5CX_API_REVISION);

		// ::::::::::::::::::: Setup :::::::::::::::::::::::::::
		status = vl53l5cx_set_resolution(&Dev, VL53L5CX_RESOLUTION_8X8);
		status = vl53l5cx_set_ranging_frequency_hz(&Dev, MTOF_FREQ);

		uint8_t current_resolution;
		vl53l5cx_get_resolution(&Dev, &current_resolution);

		int num_zones;
		int zone_wh;
		if(current_resolution == VL53L5CX_RESOLUTION_4X4)
		{ 
			num_zones = 16;
			zone_wh = 4;
		}
		else if(current_resolution == VL53L5CX_RESOLUTION_8X8)
		{ 
			num_zones = 64;
			zone_wh = 8;
		}

		// :::::::::::::::: Data acquisition loop :::::::::::::::::::::::::
		status = vl53l5cx_start_ranging(&Dev);

		while(rclcpp::ok())
		{
			status = vl53l5cx_check_data_ready(&Dev, &isReady);
			
			if(isReady)
			{
				mtofalib::msg::Mtof tof_msg;
				std::vector<float> dist_vec;
				std::vector<float> sigma_vec;
				std::vector<float> refl_vec;

				vl53l5cx_get_ranging_data(&Dev, &Results);

				for(i = 0; i < num_zones; i++)
				{
					float dist = Results.distance_mm[VL53L5CX_NB_TARGET_PER_ZONE*i];
					float sigma = Results.range_sigma_mm[VL53L5CX_NB_TARGET_PER_ZONE*i];
					float refl = Results.reflectance[VL53L5CX_NB_TARGET_PER_ZONE*i];
					dist_vec.push_back(dist);
					sigma_vec.push_back(sigma);
					refl_vec.push_back(refl);
				}
                tof_msg.header.stamp = this->now();
                if(current_resolution == VL53L5CX_RESOLUTION_4X4){ tof_msg.zone_height = 4; tof_msg.zone_width = 4; }
                else if(current_resolution == VL53L5CX_RESOLUTION_8X8){ tof_msg.zone_height = 8; tof_msg.zone_width = 8; }
                tof_msg.max_dist = 4.0;
                tof_msg.min_dist = 0.02;
                tof_msg.zone_distance_mm = dist_vec;
                tof_msg.zone_variance_mm = sigma_vec;
                tof_msg.zone_reflectance = refl_vec;
                mtof_pub->publish(tof_msg);
			}
            // ZzZzZz for 3ms
			VL53L5CX_WaitMs(&(Dev.platform), 3);
		}

		status = vl53l5cx_stop_ranging(&Dev);

		VL53L5CX_Comms_Close(p_dev);
	}
};


// :::::::::::::::::::::::::::::::::: MAIN :::::::::::::::::::::::::::::::::::::::
int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MtofPublisher>());
    rclcpp::shutdown();
    return 0;
}
