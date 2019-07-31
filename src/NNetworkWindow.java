
import javafx.application.Application;
import javafx.event.Event;
import javafx.event.EventHandler;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.layout.StackPane;
import javafx.stage.Stage;

import java.awt.event.ActionEvent;


//import com.sun.corba.se.pept.transport.EventHandler;

//import javafx.*;

public class NNetworkWindow extends Application{// implements EventHandler<ActionEvent>{

	private Stage arg;

	public void go(){
		this.launch("");
	}

	//@Override
	public void start(Stage primaryStage) throws Exception {
		primaryStage.setTitle("Hello World!");
		Button btn = new Button();
		btn.setText("Say 'Hello World'");
		//btn.setOnAction(new EventHandler<ActionEvent>() {
		//	@Override
		//	public void handle(ActionEvent event) {
		//		System.out.println("Hello World!");
		//	}
		//});

		StackPane root = new StackPane();
		root.getChildren().add(btn);
		primaryStage.setScene(new Scene(root, 300, 250));
		primaryStage.show();
		
	}

	
	
	
	
	
	
	
	
	
	
}
