
package neureka.ngui;

import neureka.core.T;
import neureka.ngui.tile.N3DSpace;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.ComponentOrientation;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;

import javax.swing.BorderFactory;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSplitPane;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.border.Border;
import javax.swing.text.BadLocationException;

public class NVControlFrame extends JFrame implements ActionListener, KeyListener, ItemListener, MouseListener, MouseWheelListener {

	JTextArea consoleField;
	JTextArea variableDisplayField;
	
	JTextField commandField;
	
	JTextField weightModField;
	JTextField shiftModField;
	JTextField connectionIndexingField;
	JTextField inputIndexingField;
	
	JTextField inputModField;
	JTextField activationModField;
	JTextField historyIndexingField;
	JTextField outputOptimumModField;
	
	JButton displayAllVariables;
	JButton displayVariables;
	JButton clearDisplay;
	JButton randomizeVariables;
	
	JButton VariableUpdateButton;
	JButton VariableResetButton;
	
	JButton varPanelButton;
	JButton IOPanelButton;
	
	JPanel checkPanel;
	JCheckBox isFirstButton;
	JCheckBox isLastButton;
	JCheckBox isFunctionalButton;
	JCheckBox isHiddenButton;
	JCheckBox isRootNodeButton;
	
	N3DSpace Graph;
	
	JComboBox RootTypeSetter;
	
	T UnitCore;
	
	boolean entered = false;
	boolean check = false;
	String input = "";
    
	boolean inputFieldEntered = false;
	boolean outputFieldEntered = false;
	boolean optimumFieldEntered = false;
	boolean historyIndexingFieldEntered = false;
	boolean inputIndexingFieldEntered = false;
	boolean weightIndexingFieldEntered =false;
	
	boolean weightFieldEntered = false;
	boolean biasFieldEntered = false;
	
	public NVControlFrame(String title, T neuron) {

		UnitCore = neuron;
		
		this.setTitle(title);
		this.setSize(600, 400);
		//this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); 
		this.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE); 
		this.setLocationRelativeTo(null);
		
		consoleField = new JTextArea();
		consoleField.setText("PROCESS:\n");
		// ausgabefeld.setBorder(BorderFactory.createLineBorder(Color.black));
		consoleField.setEditable(false);
		
		Font txt = new Font("Verdana", Font.LAYOUT_LEFT_TO_RIGHT, 15);
		//Font txt = new Font("Verdana", Font.LAYOUT_NO_LIMIT_CONTEXT, 15);
		consoleField.setBackground(Color.BLACK);
		consoleField.setForeground(Color.CYAN);
		consoleField.setFont(txt);

		JScrollPane scroll = new JScrollPane(consoleField);
		scroll.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_ALWAYS);
		scroll.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_ALWAYS);
		scroll.setBackground(Color.DARK_GRAY);
		scroll.setForeground(Color.BLUE);
		scroll.getVerticalScrollBar().setBackground(Color.DARK_GRAY);
		scroll.getHorizontalScrollBar().setBackground(Color.DARK_GRAY);

		commandField = new JTextField();	
		commandField.setText("Commands...");	
		commandField.setEditable(true);		
		commandField.setBackground(Color.BLACK);
		commandField.setForeground(Color.CYAN);
		commandField.setFont(txt);
		commandField.addKeyListener(this);
		
		JPanel panel = new JPanel();
		panel.setLayout(new BorderLayout());
		   JLabel label = new JLabel("  PROCESS OUTPUT:  ",JLabel.CENTER);
		   label.setOpaque(true);
		   label.setBackground(Color.CYAN);
		   label.setForeground(Color.BLACK);
		   Border bordery = BorderFactory.createBevelBorder(0, Color.BLACK, Color.BLACK);//createLineBorder(Color.darkGray, 5);
		   label.setBorder(bordery);
		panel.add(label, BorderLayout.NORTH);
		panel.add(scroll, BorderLayout.CENTER);
		panel.add(commandField, BorderLayout.SOUTH);
		panel.setBackground(Color.DARK_GRAY);
   	    panel.setBorder(BorderFactory.createEmptyBorder(3, 3, 3, 3));
		///this.getContentPane().e_add(inputField, BorderLayout.SOUTH);
		
		//==============================================================================================================
		JSplitPane horizontalSplitPane = new JSplitPane(JSplitPane.VERTICAL_SPLIT);
		horizontalSplitPane.setContinuousLayout(true); // Update window as splitter is moved
		   //--------------------------------------------------------------------------------------
		   Graph = new N3DSpace(78);
		   JPanel graphPanel = new JPanel();
		   graphPanel.setLayout(new BorderLayout());
		    	JLabel graphLabel = new JLabel("  PROCESS GRAPH:  ",JLabel.CENTER);
		    	graphLabel.setOpaque(true);
		    	graphLabel.setBackground(Color.CYAN);
		    	graphLabel.setForeground(Color.BLACK);
		    	Border graphLabelBorder = BorderFactory.createBevelBorder(0, Color.BLACK, Color.BLACK);//createLineBorder(Color.darkGray, 5);
		    	graphLabel.setBorder(graphLabelBorder);
		   graphPanel.add(graphLabel, BorderLayout.NORTH);
		   graphPanel.add(Graph.getSurface(), BorderLayout.CENTER);
		   graphPanel.setBackground(Color.DARK_GRAY);
     	   graphPanel.setBorder(BorderFactory.createEmptyBorder(3, 3, 3, 3));

		  //--------------------------------------------------------------------------------------
		horizontalSplitPane.setTopComponent(panel);// GRAPH IMPLEMENTATION HERE!
		horizontalSplitPane.setBottomComponent(graphPanel);
		//==============================================================================================================	
		JSplitPane verticalSplitPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT);
        verticalSplitPane.setContinuousLayout(true); // Update window as splitter is moved
        verticalSplitPane.setLeftComponent(horizontalSplitPane);
            JSplitPane rightHorizontalSplitPane = new JSplitPane(JSplitPane.VERTICAL_SPLIT);
            rightHorizontalSplitPane.setContinuousLayout(true); // Update window as splitter is moved
            
           
            //==============================================================================================
            	 variableDisplayField = new JTextArea();
            	 variableDisplayField.setText("Status display...\n");
            	 variableDisplayField.setEditable(false);
            	 variableDisplayField.setBackground(Color.BLACK);
            	 variableDisplayField.setForeground(Color.CYAN);
            	 variableDisplayField.setFont(txt);
            	 JScrollPane weightDisplayScroll = new JScrollPane(variableDisplayField);
            	 weightDisplayScroll.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_ALWAYS);
            	 weightDisplayScroll.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_ALWAYS);
            	 weightDisplayScroll.setBackground(Color.DARK_GRAY);
            	 //weightDisplayScroll.setForeground(Color.BLUE);
            	 weightDisplayScroll.getVerticalScrollBar().setBackground(Color.DARK_GRAY);
            	 weightDisplayScroll.getHorizontalScrollBar().setBackground(Color.DARK_GRAY);

            	     JPanel weightDisplayPanel = new JPanel();
            	     weightDisplayPanel.setLayout(new BorderLayout());
            	     weightDisplayPanel.setBorder(BorderFactory.createEmptyBorder(3, 3, 3, 3));
	 			   	 weightDisplayPanel.setBackground(Color.DARK_GRAY);
            	     
            	     
            	        JLabel weightDisplayLabel = new JLabel("  STATUS DISPLAY:  ",JLabel.CENTER);
            	        weightDisplayLabel.setOpaque(true);
            	        weightDisplayLabel.setBackground(Color.CYAN);
            	        weightDisplayLabel.setForeground(Color.BLACK);
     		    	    Border weightDisplayLabelBorder = BorderFactory.createBevelBorder(0, Color.BLACK, Color.BLACK);//createLineBorder(Color.darkGray, 5);
     		    	    weightDisplayLabel.setBorder(weightDisplayLabelBorder);
            	        
            	        
            	     	   JPanel weightDisplaySubPanel = new JPanel();
            	     	   weightDisplaySubPanel.setLayout(new BorderLayout());
            	     	   weightDisplaySubPanel.add(weightDisplayLabel, BorderLayout.NORTH);
            	     	   weightDisplaySubPanel.add(weightDisplayScroll, BorderLayout.CENTER);	
            	     	
            	     weightDisplayPanel.add(weightDisplaySubPanel, BorderLayout.CENTER);
            	        JPanel variableDisplaySubPanel2 = new JPanel();
            	           //variableDisplaySubPanel2.setLayout(new BorderLayout());
            	           variableDisplaySubPanel2.setLayout(new FlowLayout());
            	           variableDisplaySubPanel2.setComponentOrientation(ComponentOrientation.LEFT_TO_RIGHT);

            	               clearDisplay = new JButton("CLEAR DISPLAY");
            	               clearDisplay.addActionListener(this);
            	           variableDisplaySubPanel2.add(clearDisplay);
            	               displayAllVariables = new JButton("DISPLAY ALL!");
         	                   displayAllVariables.addActionListener(this);
         	               variableDisplaySubPanel2.add(displayAllVariables);
         	                   displayVariables = new JButton("DISPLAY!");
    	                       displayVariables.addActionListener(this);
    	                   variableDisplaySubPanel2.add(displayVariables); 
    	                       randomizeVariables = new JButton("RANDOMIZE!");
    	                       randomizeVariables.addActionListener(this);
    	                   variableDisplaySubPanel2.add(randomizeVariables);     
         	           weightDisplayPanel.add(variableDisplaySubPanel2, BorderLayout.SOUTH);
            	     //===========================================================================================
            	       JPanel VariableAccessSubPanel = new JPanel();
            	           VariableAccessSubPanel.setLayout(new BorderLayout());
            	 		   weightModField = new JTextField();	
            	 		      weightModField.setText("Weight");	
            	 		      weightModField.setEditable(true);		
            	 		      weightModField.setBackground(Color.BLACK);
            	 		      weightModField.setForeground(Color.CYAN);
            	 		      weightModField.setFont(txt);
            	 		      weightModField.addKeyListener(this);
            	 		      weightModField.addMouseListener(this);
            	 		      weightModField.addMouseWheelListener(this);
            	 		      
            	 		   shiftModField = new JTextField();	
            	 		      shiftModField.setText("Bias");	
            	 		      shiftModField.setEditable(true);		
            	 		      shiftModField.setBackground(Color.BLACK);
            	 		      shiftModField.setForeground(Color.CYAN);
            	 		      shiftModField.setFont(txt);
            	 		      shiftModField.addKeyListener(this);
            	 		      shiftModField.addMouseListener(this);
            	 		      shiftModField.addMouseWheelListener(this);
            	 		      
            	 		   connectionIndexingField = new JTextField();	
            	 			   connectionIndexingField.setText("Weight Index");	
            	 			   connectionIndexingField.setEditable(true);		
            	 			   connectionIndexingField.setBackground(Color.BLACK);
            	 			   connectionIndexingField.setForeground(Color.CYAN);
            	 			   connectionIndexingField.setFont(txt);
            	 			   connectionIndexingField.setHorizontalAlignment(JTextField.CENTER);
            	 			   connectionIndexingField.addKeyListener(this);
            	 			   connectionIndexingField.addMouseListener(this);
            	 			   connectionIndexingField.addMouseWheelListener(this);
            	 			   
            	 		   inputIndexingField = new JTextField();	
            	 			   inputIndexingField.setText("Value Index");
            	 			   inputIndexingField.setEditable(true);		
            	 			   inputIndexingField.setBackground(Color.BLACK);
            	 			   inputIndexingField.setForeground(Color.CYAN);
            	 			   inputIndexingField.setFont(txt);
            	 			   inputIndexingField.setHorizontalAlignment(JTextField.CENTER);
            	 			   inputIndexingField.addKeyListener(this);
            	 			   inputIndexingField.addMouseListener(this);
            	 			   inputIndexingField.addMouseWheelListener(this);
            	 			   
            	 			//=========================================================================================== 
            	 			   ///////////////////////////////////
                	 		   inputModField = new JTextField();	
                	 		      inputModField.setText("Value");
                	 		      inputModField.setEditable(true);		
                	 		      inputModField.setBackground(Color.BLACK);
                	 		      inputModField.setForeground(Color.CYAN);
                	 		      inputModField.setFont(txt);
                	 		      inputModField.addKeyListener(this);
                	 		      inputModField.addMouseListener(this);
                	 		      inputModField.addMouseWheelListener(this);
                	 		      
                	 		   activationModField = new JTextField();	
                	 		      activationModField.setText("Activation");	
                	 		      activationModField.setEditable(true);		
                	 		      activationModField.setBackground(Color.BLACK);
                	 		      activationModField.setForeground(Color.CYAN);
                	 		      activationModField.setFont(txt);
                	 		      activationModField.addKeyListener(this);
                	 		      activationModField.addMouseListener(this);
                	 		      activationModField.addMouseWheelListener(this);
                	 		      
                	 		   historyIndexingField = new JTextField();	
                	 			   historyIndexingField.setText("History Index");	
                	 			   historyIndexingField.setEditable(true);		
                	 			   historyIndexingField.setBackground(Color.BLACK);
                	 			   historyIndexingField.setForeground(Color.CYAN);
                	 			   historyIndexingField.setFont(txt);
                	 			   historyIndexingField.setHorizontalAlignment(JTextField.CENTER);
                	 			   historyIndexingField.addKeyListener(this);
                	 			   historyIndexingField.addMouseListener(this);
                	 		       historyIndexingField.addMouseWheelListener(this);
                	 			   
                	 		   outputOptimumModField = new JTextField();	
                	 			   outputOptimumModField.setText("Optimum");	
                	 			   outputOptimumModField.setEditable(true);		
                	 			   outputOptimumModField.setBackground(Color.BLACK);
                	 			   outputOptimumModField.setForeground(Color.CYAN);
                	 			   outputOptimumModField.setFont(txt);  
                	 			   outputOptimumModField.addKeyListener(this);
                	 			   outputOptimumModField.addMouseWheelListener(this);
                	 			   outputOptimumModField.addMouseListener(this);
            	 			///////////////////////////////////////////////////////////
                	 			   
            	 			   	JPanel indexingSubPanel = new JPanel();
            	 			   		indexingSubPanel.setLayout(new BorderLayout());
            	 			   		indexingSubPanel.add(new JLabel("Indexing:",JLabel.CENTER), BorderLayout.NORTH);
            	 			   	    //indexingSubPanel.setBackground(Color.DARK_GRAY);
            	 	     	        indexingSubPanel.setBorder(BorderFactory.createEmptyBorder(0, 3, 0, 0));
            	 			   		//indexingSubPanel.e_add(shiftIndexingField, BorderLayout.CENTER);
            	 			   		//indexingSubPanel.e_add(weightIndexingField, BorderLayout.SOUTH);
            	 			   			JPanel indexingModPanel = new JPanel();
            	 			   				indexingModPanel.setLayout(new BoxLayout(indexingModPanel, BoxLayout.PAGE_AXIS));
            	 			   				indexingModPanel.add(inputIndexingField);
            	 			   				indexingModPanel.add(connectionIndexingField);
            	 			   				indexingModPanel.add(historyIndexingField);
            	 			   		indexingSubPanel.add(indexingModPanel, BorderLayout.CENTER);	
            	 			   				
            	 			   	JPanel valueSubPanel = new JPanel();
            	 			   		valueSubPanel.setLayout(new BorderLayout());
            	 			   		valueSubPanel.add(new JLabel("Values:",JLabel.CENTER), BorderLayout.NORTH);
            	 			   		//valueSubPanel.e_add(shiftModField, BorderLayout.CENTER);
            	 			   		//valueSubPanel.e_add(weightModField, BorderLayout.SOUTH);
            	 			   			JPanel valueModPanel = new JPanel();
            	 			   				valueModPanel.setLayout(new BoxLayout(valueModPanel, BoxLayout.PAGE_AXIS));
            	 			   				valueModPanel.add(shiftModField);
            	 			   				valueModPanel.add(weightModField);
            	 			   				
            	 			   				valueModPanel.add(inputModField);
            	 			   				valueModPanel.add(activationModField);
            	 			   				valueModPanel.add(outputOptimumModField);
            	 			   				
            	 			   		valueSubPanel.add(valueModPanel, BorderLayout.CENTER);
            	 			   			
            	 			   	JLabel variableAccessLabel = new JLabel("  Input access and modification:  ",JLabel.CENTER);
            	 			   		variableAccessLabel.setOpaque(true);
            	 			   		variableAccessLabel.setBackground(Color.CYAN);
            	 			   		variableAccessLabel.setForeground(Color.BLACK);
            	 			   		Border border = BorderFactory.createBevelBorder(0, Color.BLACK, Color.BLACK);//createLineBorder(Color.darkGray, 5);
            	 			   		variableAccessLabel.setBorder(border);

            	 			   		VariableAccessSubPanel.add(variableAccessLabel, BorderLayout.NORTH);
            	 			   		VariableAccessSubPanel.add(indexingSubPanel, BorderLayout.WEST);
            	 			   		VariableAccessSubPanel.add(valueSubPanel, BorderLayout.CENTER);
            	 			   		VariableAccessSubPanel.setBorder(BorderFactory.createEmptyBorder(3, 3, 3, 3));
            	 			   		VariableAccessSubPanel.setBackground(Color.DARK_GRAY);
            	 			   		
            	 			   		JPanel VariableAccessPanel = new JPanel();
            	 			   		VariableAccessPanel.setLayout(new BorderLayout());
            	 			   		   JLabel weightControlLabel = new JLabel("  Actions:  ",JLabel.CENTER);
            	 			   		   //weightControlLabel.setOpaque(true);
            	 			   		   //weightControlLabel.setBackground(Color.CYAN);
            	 			   		   //weightControlLabel.setForeground(Color.BLACK);
            	 			   		   //Border bordero = BorderFactory.createBevelBorder(0, Color.BLACK, Color.BLACK);//createLineBorder(Color.darkGray, 5);
            	 			   		   //weightControlLabel.setBorder(bordero);
            	 			   
            	 			   		VariableAccessPanel.add(weightControlLabel, BorderLayout.NORTH);
            	 			   	
            	 			   			JPanel VariableActionButtonPanel = new JPanel();
            	 			   				VariableActionButtonPanel.
            	 			   				setLayout(new BoxLayout(VariableActionButtonPanel, BoxLayout.Y_AXIS));
            	 			   				
            	 			   			//VariableActionButtonPanel.setLayout(new BorderLayout());
            	 			   		    //VariableActionButtonPanel.setLayout(new GridBagLayout());
            	 			   			//	GridBagConstraints cons = new GridBagConstraints();
            	 			   			//	cons.fill = GridBagConstraints.HORIZONTAL;
            	 			   			//    //cons.fill = GridBagConstraints.VERTICAL;
            	 			   			//	cons.weightx = 1;
            	 			   			//	cons.gridx = 0;
            	 			   				VariableUpdateButton = new JButton("UPDATE");
            	 			   	            VariableUpdateButton.addActionListener(this);
            	 			   	            
            	 			   				VariableResetButton = new JButton("RESET");
            	 			   			    VariableResetButton.addActionListener(this);
            	 			   			//	VariableActionButtonPanel.e_add(VariableUpdateButton, cons);
            	 			   		    //	VariableActionButtonPanel.e_add(VariableResetButton, cons);
            	 			   			  //  VariableActionButtonPanel.e_add(VariableUpdateButton, BorderLayout.NORTH);
                	 			   			//VariableActionButtonPanel.e_add(VariableResetButton, BorderLayout.SOUTH);
                	 			   				JPanel updateContainer = new JPanel();
                	 			   				updateContainer.setLayout(new BorderLayout());
                	 			   				updateContainer.add(VariableUpdateButton, BorderLayout.CENTER);
                	 			   				updateContainer.setBorder(BorderFactory.createEmptyBorder(5, 5, 0, 5));
                	 			   				
            	 			     			VariableActionButtonPanel.add(updateContainer);
            	 			     			
            	 			     				JPanel resetContainer = new JPanel();
            	 			     				resetContainer.setLayout(new BorderLayout());
            	 			     				resetContainer.add(VariableResetButton, BorderLayout.CENTER);
            	 			     				resetContainer.setBorder(BorderFactory.createEmptyBorder(0, 5, 5, 5));
            	 			     				
            	 			   	            VariableActionButtonPanel.add(resetContainer);  
            	 			     				
            	 			      VariableAccessPanel.add(VariableActionButtonPanel, BorderLayout.CENTER);
            	 			   
            	 			   VariableAccessSubPanel.add(VariableAccessPanel,BorderLayout.EAST) ;  
            	 			  
            	 			   //=============================================================================================
            	 			   
            	 				JSplitPane IndexingSplitPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT);
            	 				IndexingSplitPane.setContinuousLayout(true); // Update window as splitter is moved
            	 				
            	 				//In initialization code:
            	 				isFirstButton = new JCheckBox("is first layer");
            	 			    isFirstButton.setMnemonic(KeyEvent.VK_F); 
            	 			    isFirstButton.setSelected(UnitCore.rqsGradient());

            	 			    isHiddenButton = new JCheckBox("is hidden layer");
            	 			    isHiddenButton.setMnemonic(KeyEvent.VK_H); 
            	 			    isHiddenButton.setSelected(UnitCore.rqsGradient());

            	 			    isLastButton = new JCheckBox("is last layer");
            	 			    isLastButton.setMnemonic(KeyEvent.VK_L); 
            	 			    isLastButton.setSelected(UnitCore.rqsGradient());

            	 			    isFunctionalButton = new JCheckBox("is paralyzed");
            	 			    isFunctionalButton.setMnemonic(KeyEvent.VK_X); 
            	 			    isFunctionalButton.setSelected(UnitCore.rqsGradient());
            	 			    
            	 			    isRootNodeButton = new JCheckBox("is root node");
            	 			    isRootNodeButton.setMnemonic(KeyEvent.VK_C); 
            	 			    isRootNodeButton.setSelected(UnitCore.rqsGradient());
            	 			    
            	 			     String[] choices = {
            	 			    "Basic root node","Memory root node",
            	 			    "Value root node","Super root node"
            	 			    };
            	 			    RootTypeSetter = new JComboBox(choices);
            	 			   
            	 			    //Register a listener for the check boxes.
            	 			    isFirstButton.addItemListener(this);
            	 			    isHiddenButton.addItemListener(this);
            	 			    isLastButton.addItemListener(this);
            	 			    isFunctionalButton.addItemListener(this);
            	 			    isRootNodeButton.addItemListener(this);

            	 			    RootTypeSetter.addItemListener(this);
            	 			    
            	 			    checkPanel = new JPanel(new GridLayout(0, 1));
            	 			    checkPanel.add(isFirstButton);
            	 			    checkPanel.add(isHiddenButton);
            	 			    checkPanel.add(isLastButton);
            	 			    checkPanel.add(isFunctionalButton);
            	 			    checkPanel.add(isRootNodeButton);
            	 			    checkPanel.add(RootTypeSetter); 
            	 			    
            	 			    //!!!!!!!!!!!!!!!
            	 			    JPanel CheckPanelFlow = new JPanel();
            	 			        CheckPanelFlow.setLayout(new FlowLayout());
                	                CheckPanelFlow.setComponentOrientation(ComponentOrientation.LEFT_TO_RIGHT);
                	                CheckPanelFlow.add(checkPanel);
                	                
            	            	 JScrollPane CheckPanelScroll = new JScrollPane(CheckPanelFlow);
            	            	 CheckPanelScroll.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_ALWAYS);
            	            	 CheckPanelScroll.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_ALWAYS);
            	            	 CheckPanelScroll.setBackground(Color.DARK_GRAY);
            	            	 CheckPanelScroll.setForeground(Color.BLUE);
            	            	 CheckPanelScroll.getVerticalScrollBar().setBackground(Color.DARK_GRAY);
            	            	 CheckPanelScroll.getHorizontalScrollBar().setBackground(Color.DARK_GRAY);

            	            	     
            	            	        JLabel checkLabel = new JLabel("  SETTINGS:  ",JLabel.CENTER);
            	            	        checkLabel.setOpaque(true);
            	            	        checkLabel.setBackground(Color.CYAN);
            	            	        checkLabel.setForeground(Color.BLACK);
            	     		    	    Border CheckLabelBorder = BorderFactory.createBevelBorder(0, Color.BLACK, Color.BLACK);//createLineBorder(Color.darkGray, 5);
            	     		    	    checkLabel.setBorder(CheckLabelBorder);
            	            	        
            	            	        
            	            	     	   JPanel CheckSubPanel = new JPanel();
            	            	     	   CheckSubPanel.setLayout(new BorderLayout());
            	            	     	   CheckSubPanel.add(checkLabel, BorderLayout.NORTH);
            	            	     	   CheckSubPanel.add(CheckPanelScroll, BorderLayout.CENTER);	
            	            	     	   
            	            	     	   CheckSubPanel.setBackground(Color.DARK_GRAY);
            	            	     	   CheckSubPanel.setBorder(BorderFactory.createEmptyBorder(3, 3, 3, 3));
            	            	     	   
            	 			    IndexingSplitPane.setLeftComponent(CheckSubPanel);//checkPanel
            	 			    IndexingSplitPane.setRightComponent(VariableAccessSubPanel);      
            	 			
          
            	 	       rightHorizontalSplitPane.setTopComponent(IndexingSplitPane);	    
            	 			    
             rightHorizontalSplitPane.setBottomComponent(weightDisplayPanel);
        verticalSplitPane.setRightComponent(rightHorizontalSplitPane);
        //==============================================================================================================
        this.getContentPane().add(verticalSplitPane, BorderLayout.CENTER);
		
		Thread t = new Thread(()->{
			while(true) {Graph.getSurface().updateAndRedraw();
			try {
				Thread.sleep(10);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}}
		});
		t.start();
		this.setVisible(true);
	}
	
	
	public void itemStateChanged(ItemEvent e) 
	{
		Object source = e.getItemSelectable();
		if (source == isFirstButton) 
		{
			if(isFirstButton.isSelected()) 
			{if(isHiddenButton.isSelected()) {isHiddenButton.setSelected(false);}}
			//UnitCore.setFirst(isFirstButton.isSelected());
		} 
		else if (source == isHiddenButton) 
		{
    	
    	if(isHiddenButton.isSelected()) 
    	    {
    			if(isFirstButton.isSelected()) {isFirstButton.setSelected(false);}
    			if(isLastButton.isSelected())  {isLastButton.setSelected(false);}
    		}
    		//UnitCore.setHidden(isHiddenButton.isSelected());
		} 
		else if (source == isLastButton) 
		{
	    	if(isLastButton.isSelected()) {if(isHiddenButton.isSelected()) {isHiddenButton.setSelected(false);}}
	    	//UnitCore.setLast(isLastButton.isSelected());
    	
		}else if (source == isFunctionalButton) 
		{
	    	//UnitCore.setParalyzed(isFunctionalButton.isSelected());
		} 
		else if (source == isRootNodeButton) 
		{
			System.out.println("RootNode!");
    	//UnitCore.setIsRootUnit(isRootNodeButton.isSelected());
    	}

		if (e.getStateChange() == ItemEvent.DESELECTED)
		{}
  
	}
	
	
	@Override
	public void keyPressed(KeyEvent e) {
		// TODO Auto-generated method stub
		
		if(e.getSource()==commandField) {if(e.getKeyChar()=='\n') {System.out.println("Command sent!");}}
		
		if(e.getSource()==connectionIndexingField) {if(e.getKeyChar()=='\n') {System.out.println("connection index sent!");}}
		
		if(e.getSource()==inputIndexingField) {if(e.getKeyChar()=='\n') {System.out.println("input index sent!");
		
			if(isInteger(inputIndexingField.getText())) {
				int newIndex = parseInteger(inputIndexingField.getText()); System.out.println("New history index: "+newIndex);
				//shiftModField.setText("Bias: "+(String.valueOf(UnitCore.getBias(newIndex))));
				int newWeightIndex = parseInteger(connectionIndexingField.getText());
				//weightModField.setText("Weight: "+(String.valueOf(UnitCore.getWeight(newIndex, newWeightIndex))));
		    	}
		
			else
			{
				System.out.println("Index not acceptable!");
			}
		
		
		}}
		
		if(e.getSource()==weightModField) {if(e.getKeyChar()=='\n') {System.out.println("Weight modification sent!");}}
		
		if(e.getSource()==shiftModField) {if(e.getKeyChar()=='\n') {System.out.println("Bias modification sent!");}}
		//======================================================================================================
		
		
		if(e.getSource()==historyIndexingField) {if(e.getKeyChar()=='\n') {System.out.println("Value index sent!");
		if(isInteger(historyIndexingField.getText())) {
		int newIndex = parseInteger(historyIndexingField.getText()); System.out.println("New input index: "+newIndex);
		//inputModField.setText((String.valueOf(UnitCore.getInput(0,newIndex))));
		}
		else
		{
			System.out.println("Index not acceptable!");
		}
		
		}
		
		}

		
		if(e.getSource()==inputModField) {if(e.getKeyChar()=='\n') {System.out.println("Value modification sent!");}}
		//=====================================================================================================
		if(e.getSource()==activationModField) {if(e.getKeyChar()=='\n') {System.out.println("Output modification sent!");}}

		if(e.getSource()==outputOptimumModField) {if(e.getKeyChar()=='\n') {System.out.println("Output optimum modification sent!");}}
		//======================================================================================================
		
		
	}

	public void actionPerformed(ActionEvent e) {
        if(e.getSource() == VariableResetButton) {System.out.println("I/O update button pressed!");
        if(isInteger(historyIndexingField.getText())) {// double
        	try
        	{
        	  int newIndex = Integer.parseInt(historyIndexingField.getText());
        		
        	    
        	  double newValue = Double.parseDouble(inputModField.getText()); System.out.println("New input: "+newValue);
    		    
        	    
        	    //inputModField.setText(""+UnitCore.getInput(0,newIndex));
        	    
        		
        		//outputOptimumModField.setText(""+UnitCore.getOptimum(0));
        		
                //newValue = Double.parseDouble(outputOptimumModField.getText()); System.out.println("New output optimum: "+newValue);
        		
        		//Neuron.setOutput..
        		
        	}
        	catch(NumberFormatException ne)
        	{   
        	  //not a double
        	}
        	
        	
        	
        	
    	  }
        
        }
        
        if(e.getSource() == VariableUpdateButton) {System.out.println("Value mod update button pressed!");
        if(isInteger(historyIndexingField.getText())) {// double
        	try
        	{
        	    int newIndex = Integer.parseInt(historyIndexingField.getText());
        		double newValue = Double.parseDouble(inputModField.getText()); 
        		System.out.println("New input: "+newValue);
    		    
        		//UnitCore.setInput(0, newIndex, newValue);
        		
        	}
        	catch(NumberFormatException ne)
        	{   
        	  //not a double
        	}
        	
        	
        	
        	
    	  }
        
        }
        
        if(e.getSource() == VariableUpdateButton) {System.out.println("Optimum mod update button pressed!");
    
        	try
        	{
        	  //int newIndex = Integer.parseInt(inputIndexingField.getText());
        		double newValue = Double.parseDouble(outputOptimumModField.getText()); System.out.println("New output optimum: "+newValue);
    		    
        		//Neuron.setInput(newIndex, newValue);
        		//UnitCore.setOptimum(0,newValue);
        		
        	}
        	catch(NumberFormatException ne)
        	{   
        	  //not a double
        	}

        
        }
        
        
        if(e.getSource() == VariableUpdateButton) {System.out.println("Weight/Bias update button pressed!");}
        
        if(e.getSource() == clearDisplay) {System.out.println("Weight display clear button pressed!"); clearDisplay();}

        if(e.getSource() == displayAllVariables) {System.out.println("Display all button pressed!"); displayAllVariables();}
        
        if(e.getSource() == displayVariables) {System.out.println("Display button pressed!");displayVariables();}
        
        if(e.getSource() == randomizeVariables) {System.out.println("Randomize button pressed!");
        //UnitCore.randomizeBiasAndWeight();
        }
      
    }  
	
	
	private void displayAllVariables()
	{	
		display(UnitCore.toString());	
	}
	
	private void displayVariables()
	{
		double[][] weights = {UnitCore.value()};
		displayLine("Node state: ");
		if(weights==null) {display("Node has no connections. \n"); return;}
		display("\n"); 
		 for(int Ii=0; Ii<weights.length;Ii++) 
		     {
		     	//displayLine("Bias["+Ii+"]:("+UnitCore.getBias(Ii)+");");
		      if(weights[Ii]!=null) {displayLine("-displayCurrentState|: AIC["+Ii+"].length:("+weights[Ii].length+")");}else {displayLine("-displayCurrentState|: AIC["+Ii+"] is null!");}
		      if(!UnitCore.rqsGradient()&&weights[Ii]!=null)
		         {display("Weight["+Ii+"][0-"+(weights[Ii].length-1)+"]:( ");
		          for(int Ni=0; Ni<weights[Ii].length; Ni++) {display("["+Ni+"]:("+weights[Ii][Ni]+"), ");}
		      display(");\n");}
		     }
		 display("\n");		
	}
	
	private void displayLine(String text)
	{
		
		variableDisplayField.setText(variableDisplayField.getText() + text + "\n");
		if (variableDisplayField.getLineCount() > 100) {
			try {
				variableDisplayField.replaceRange("", 0, variableDisplayField.getLineEndOffset(0));
			} catch (BadLocationException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
	}
	private void display(String text)
	{
		variableDisplayField.setText(variableDisplayField.getText() + text);
		if (variableDisplayField.getLineCount() > 100) {
			try {
				variableDisplayField.replaceRange("", 0, variableDisplayField.getLineEndOffset(0));
			} catch (BadLocationException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	
	}
	private void clearDisplay()
	{
		
		variableDisplayField.setText("");
	}
	
	
	public synchronized void println(String text) {
		consoleField.setText(consoleField.getText() + text + "\n");
		if (consoleField.getLineCount() > 100) {
			try {
				consoleField.replaceRange("", 0, consoleField.getLineEndOffset(0));
			} catch (BadLocationException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

	
	public synchronized void print(String text) {
		consoleField.setText(consoleField.getText() + text);
		if (consoleField.getLineCount() > 100) {
			try {
				consoleField.replaceRange("", 0, consoleField.getLineEndOffset(0));
			} catch (BadLocationException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

	public String read() {
		entered = false;

		while (!entered) {
			try {
				Thread.sleep(100);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		return input;
	}




	@Override
	public void keyReleased(KeyEvent e) {
		// TODO Auto-generated method stub
		
	}


	@Override
	public void keyTyped(KeyEvent e) {
		// TODO Auto-generated method stub
		
	}




	@Override
	public void mouseClicked(MouseEvent arg0) {
		// TODO Auto-generated method stub
		
	}




	@Override
	public void mouseEntered(MouseEvent e) {
		// TODO Auto-generated method stub
		
		System.out.println("entered");

		if(weightModField==e.getSource()) {weightFieldEntered=true; biasFieldEntered=false; weightIndexingFieldEntered=false; inputIndexingFieldEntered = false; historyIndexingFieldEntered = false;
		optimumFieldEntered = false; outputFieldEntered = false; inputFieldEntered = false;}
		
		if(shiftModField==e.getSource()) {weightFieldEntered=false; biasFieldEntered=true; weightIndexingFieldEntered=false; inputIndexingFieldEntered = false; historyIndexingFieldEntered = false;
        optimumFieldEntered = false; outputFieldEntered = false; inputFieldEntered = false;}

		if(connectionIndexingField==e.getSource()) {weightFieldEntered=false; biasFieldEntered=false; weightIndexingFieldEntered=true; inputIndexingFieldEntered = false; historyIndexingFieldEntered = false;
        optimumFieldEntered = false; outputFieldEntered = false; inputFieldEntered = false;}

		if(inputIndexingField==e.getSource()) {weightFieldEntered=false; biasFieldEntered=false; weightIndexingFieldEntered=false; inputIndexingFieldEntered = true; historyIndexingFieldEntered = false;
        optimumFieldEntered = false; outputFieldEntered = false; inputFieldEntered = false;}
		
		if(inputModField==e.getSource()) {weightFieldEntered=false; biasFieldEntered=false; weightIndexingFieldEntered=false; inputIndexingFieldEntered = false; historyIndexingFieldEntered = false;
        optimumFieldEntered = false; outputFieldEntered = false; inputFieldEntered = true;}

		if(activationModField==e.getSource()) {weightFieldEntered=false; biasFieldEntered=false; weightIndexingFieldEntered=false; inputIndexingFieldEntered = false; historyIndexingFieldEntered = false;
        optimumFieldEntered = false; outputFieldEntered = true; inputFieldEntered = false;}

		if(historyIndexingField==e.getSource()) {weightFieldEntered=false; biasFieldEntered=false; weightIndexingFieldEntered=false; inputIndexingFieldEntered = false; historyIndexingFieldEntered = true;
        optimumFieldEntered = false; outputFieldEntered = false; inputFieldEntered = false;}
		
		if(outputOptimumModField==e.getSource()) {weightFieldEntered=false; biasFieldEntered=false; weightIndexingFieldEntered=false; inputIndexingFieldEntered = false; historyIndexingFieldEntered = false;
        optimumFieldEntered = true; outputFieldEntered = false; inputFieldEntered = false;}
		
	}




	@Override
	public void mouseExited(MouseEvent e) {
		// TODO Auto-generated method stub
		
		
		// TODO Auto-generated method stub
		if(weightModField==e.getSource()) {setFieldEnteredFalse();}

		if(shiftModField==e.getSource()) {setFieldEnteredFalse();}

		if(connectionIndexingField==e.getSource()) {setFieldEnteredFalse();}
		
		if(inputIndexingField==e.getSource()) {setFieldEnteredFalse();}

		if(inputModField==e.getSource()) {setFieldEnteredFalse();}

		if(activationModField==e.getSource()) {setFieldEnteredFalse();}
		
		if(historyIndexingField==e.getSource()) {setFieldEnteredFalse();}

		if(outputOptimumModField==e.getSource()) {setFieldEnteredFalse();}

		
	}


 private void setFieldEnteredFalse() {
	 weightFieldEntered=false; biasFieldEntered=false; weightIndexingFieldEntered=false; inputIndexingFieldEntered = false; historyIndexingFieldEntered = false;
     optimumFieldEntered = false; outputFieldEntered = false; inputFieldEntered = false;
 }
	
	



	@Override
	public void mousePressed(MouseEvent arg0) {
		// TODO Auto-generated method stub
		
	}




	@Override
	public void mouseReleased(MouseEvent arg0) {
		// TODO Auto-generated method stub
		
	}




	@Override
	public void mouseWheelMoved(MouseWheelEvent e) {
		// TODO Auto-generated method stub
		
		if(weightFieldEntered) {
			
			System.out.println("optaining weight");
		    if(isInteger(weightModField.getText())) 
		       { int newValue = parseInteger(weightModField.getText()); System.out.println("New weight: "+newValue);
			    //workload=newWorkload;
		         newValue+=e.getWheelRotation();
		         weightModField.setText("Weight: "+newValue);
		       }
			    else
			   {System.out.println("Weight mod not acceptable!");
			   weightModField.setText("---");
			   }		
			
		}
		
		if(biasFieldEntered) {
			
			System.out.println("optaining Bias");
		    if(isInteger(shiftModField.getText())) 
		       { int newValue = parseInteger(shiftModField.getText()); System.out.println("New bias: "+newValue);
			    //workload=newWorkload;
		         newValue+=e.getWheelRotation();
		         shiftModField.setText("Bias: "+newValue);
		       }
			    else
			   {System.out.println("Bias mod not acceptable!");
			   shiftModField.setText("0");
			   }
			
			
		}
		
		if(weightIndexingFieldEntered) {
			
			System.out.println("optaining weight index.");
		    if(isInteger(connectionIndexingField.getText())) 
		       { int newValue = parseInteger(connectionIndexingField.getText()); System.out.println("New weight indexing: "+newValue);
			    //workload=newWorkload;
		         newValue+=e.getWheelRotation();
		         connectionIndexingField.setText("Wi: "+newValue);
		       }
			    else
			   {System.out.println("Weight index not acceptable!");
			   connectionIndexingField.setText("0");
			   }
			
		}
		
		if(inputIndexingFieldEntered) {
			
			System.out.println("optaining input index.");
		    if(isInteger(inputIndexingField.getText())) 
		       {
		    	int newIndex = parseInteger(inputIndexingField.getText()); System.out.println("New input index: "+newIndex);
			    
		        newIndex+=e.getWheelRotation();
		        inputIndexingField.setText("Ii: "+newIndex);
		         
		        newIndex = Math.abs(newIndex);
		        
				//shiftModField.setText("Bias: "+(String.valueOf(UnitCore.getBias(newIndex))));
				
				int newWeightIndex = parseInteger(connectionIndexingField.getText());
				//weightModField.setText("Weight: "+(String.valueOf(UnitCore.getWeight(newIndex, newWeightIndex))));
		       }
			    else
			   {
			    	System.out.println("Value index mod not acceptable!");
			    	inputIndexingField.setText("0");
			   }
		}
		
		if(historyIndexingFieldEntered) {
			
			System.out.println("optaining history index.");
		    if(isInteger(historyIndexingField.getText())) 
		       { int newIndex = parseInteger(historyIndexingField.getText()); System.out.println("New history index: "+newIndex);
			    //workload=newWorkload;
		         newIndex+=e.getWheelRotation();
		         historyIndexingField.setText("Hi: "+newIndex);
		         newIndex = Math.abs(newIndex);
		         //getting gradients and input at index...

		         if(UnitCore!=null) {
		        	 ////NVMemory Memory = (NVMemory) UnitCore.asCore().findModule(NVMemory.class);
		        	 //if(Memory!=null) {
		        	//	//double[][] activation = Memory.getActivationAt(Memory.activeTimeDeltaWithin(new BigInteger(newIndex+"")));
		        	//	if(activation!=null) {
		        	//		activationModField.setText("Activation: "+activation[0]);
		        	//		if(activation.length>=3) {outputOptimumModField.setText("Optimum: "+activation[0][2]);}
		        	//	}
		        	 //}
		         }
		       }
			    else
			   {
			    	System.out.println("Value index not acceptable!");
			    	historyIndexingField.setText("---");
			   }
			
		}
		
		if(optimumFieldEntered) {
			
			System.out.println("optaining optimum...");
		    if(isInteger(outputOptimumModField.getText())) 
		       { int newValue = parseInteger(outputOptimumModField.getText()); System.out.println("New optimum: "+newValue);
			    //workload=newWorkload;
		         newValue+=e.getWheelRotation();
		         outputOptimumModField.setText("Optimum: "+newValue);
		       }
			    else
			   {System.out.println("Output optimum not acceptable!");
			   outputOptimumModField.setText("---");
			   }
		}
		
		if(outputFieldEntered) {
			System.out.println("optaining output mod...");
		    if(isInteger(activationModField.getText())) 
		       { int newValue = parseInteger(activationModField.getText()); System.out.println("New output: "+newValue);
			    //workload=newWorkload;
		         newValue+=e.getWheelRotation();
		         activationModField.setText("Activation: "+newValue);
		       }
			    else
			   {System.out.println("Output mod not acceptable!");
			   activationModField.setText("---");
			   }
		}
		
		if(inputFieldEntered) {
			System.out.println("optaining input...");
		    if(isInteger(inputModField.getText())) 
		       { int newValue = parseInteger(inputModField.getText()); System.out.println("New input: "+newValue);
		         newValue+=e.getWheelRotation();
		         inputModField.setText("Value: "+newValue);
		       }
			    else
			   {System.out.println("Value mod not acceptable!");
			   inputModField.setText("---");
			   }
		}
	}
	
	
	
	private int parseInteger(String S) {
		char sign = ' ';
		int result=0;
		for(int i=0; i<S.length(); i++) {
			if(S.charAt(i)=='0') {result*=10; result+=0;}
			if(S.charAt(i)=='1') {result*=10; result+=1;}
			if(S.charAt(i)=='2') {result*=10; result+=2;}
			if(S.charAt(i)=='3') {result*=10; result+=3;}
			if(S.charAt(i)=='4') {result*=10; result+=4;}
			if(S.charAt(i)=='5') {result*=10; result+=5;}
			if(S.charAt(i)=='6') {result*=10; result+=6;}
			if(S.charAt(i)=='7') {result*=10; result+=7;}
			if(S.charAt(i)=='8') {result*=10; result+=8;}
			if(S.charAt(i)=='9') {result*=10; result+=9;}
			if(S.charAt(i)=='-') {sign='-';}
		}
		if(sign=='-') {result*=-1;}
		return result;
		
	}
	
	public static boolean isInteger(String str) {
	    if (str == null) {
	        return false;
	    }
	    int length = str.length();
	    if (length == 0) {
	        return false;
	    }
	    int i = 0;
	    if (str.charAt(0) == '-') {
	        if (length == 1) {
	            return false;
	        }
	        i = 1;
	    }
	    for (; i < length; i++) {
	        char c = str.charAt(i);
	        if (c > '0' || c < '9') {
	            return true;
	        }
	    }
	    return true;
	}
	
	
	
	
}
