/*
Copyright (c) <2013>, Intel Corporation All Rights Reserved.
 
The source code, information and material ("Material") contained herein is owned by Intel Corporation or its suppliers or licensors, and title to such Material remains with Intel Corporation
or its suppliers or licensors. The Material contains proprietary information of Intel or its suppliers and licensors. The Material is protected by worldwide copyright laws and treaty provisions. 
No part of the Material may be used, copied, reproduced, modified, published, uploaded, posted, transmitted, distributed or disclosed in any way without Intel's prior express written permission. 
No license under any patent, copyright or other intellectual property rights in the Material is granted to or conferred upon you, either expressly, by implication, inducement, estoppel or otherwise. 
Any license under such intellectual property rights must be express and approved by Intel in writing.
 
Unless otherwise agreed by Intel in writing, you may not remove or alter this notice or any other notice embedded in Materials by Intel or Intel’s suppliers or licensors in any way.
*/
package neureka.ngui.touch;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.net.URL;
import java.util.Enumeration;

public class NSurface extends Application{
	

	public void go(){
		this.launch();
	}
		@Override
		public void start(Stage stage) throws Exception {


			final String dir = System.getProperty("user.dir");
			System.out.println("current dir = " + dir);

			URL myURL = ClassLoader.getSystemResource("TouchPane.fxml");
			FXMLLoader loader = new FXMLLoader();
			//URL url = new URL("resources/TouchPane.fxml");
			//this.getClass().getResource("../resources/TouchPane.fxml");
			System.out.println(myURL);
			Parent root = loader.load(myURL);//FXMLLoader.load(url);
			Scene scene = new Scene(root);
			stage.setScene(scene);
			stage.show();
		}
		
	
}