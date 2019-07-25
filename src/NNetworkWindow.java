
import javafx.application.Application;
import javafx.stage.Stage;

import java.awt.event.ActionEvent;

//import com.sun.corba.se.pept.transport.EventHandler;

//import javafx.*;

public class NNetworkWindow extends Application{// implements EventHandler<ActionEvent>{

	private Stage arg;

	//@Override
	public void start(Stage arg0) throws Exception {
		arg = arg0;
		new Thread(()->{
			arg.close();
		});

		// TODO Auto-generated method stub
		
	}

	
	
	
	
	
	
	
	
	
	
}
