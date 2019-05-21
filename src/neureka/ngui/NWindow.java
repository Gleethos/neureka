package neureka.ngui;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.ComponentOrientation;
import java.awt.FlowLayout;
import java.awt.event.ComponentEvent;
import java.awt.event.ComponentListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JPanel;
import javax.swing.JTextField;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;



public class NWindow implements ActionListener, KeyListener, MouseListener, MouseWheelListener{
	
	private static JFrame window;
	private JTextField historyScopeField;
	private JTextField currentStepField;
	private JTextField workloadInputField;
	private int step;
	private int workload;
	private int historyScope;
	private boolean stepFieldEntered=false;
	private boolean workloadFieldEntered=false;
	private boolean historyScopeFieldEntered=false;
	
	private boolean entered = false;
	private boolean check = false;

	private double lastTime=0;
	
	public static final int WIDTH = 800;
	public static final int HEIGHT = 700;
	
	//private NPanel NodeGraphPanel;
	NGraphBuilder GraphBuilder;

	private JButton startButton;
	private JButton randomizeNetworkButton;
		
	
	public NWindow()
	{
		//fullscreen
		/*
		frame.setExtendedState(JFrame.MAXIMIZED_BOTH); 
		frame.setUndecorated(true);
		frame.setVisible(true);
		*/
		GraphBuilder = new NGraphBuilder();
		
		System.setProperty("sun.java2d.opengl", "true");


		//GraphBuilder.getSurface().setPreferredSize(new Dimension(200,30));
		//GraphBuilder.getSurface().setBackground(Color.black);

		window = new JFrame("Network Display");
		window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		window.setSize(WIDTH, HEIGHT);
		window.setLocationRelativeTo(null);
		window.setResizable(true);
		//window.setLayout(null);
		window.addKeyListener(this);
		window.setFocusable(true);
		window.setUndecorated(false);
		//window.getRootPane().setWindowDecorationStyle(JRootPane.BOTTOM_ALIGNMENT);
		//((JComponent) window).setBorder(new EmptyBorder(5, 5, 5, 5));
		 // NodeGraphPanel.addKeyListener(this);
		 // NodeGraphPanel.setFocusable(true);
		  
		//window.setExtendedState(JFrame.MAXIMIZED_BOTH); 
		//window.setUndecorated(true);
		window.setVisible(true);
		
		JMenuBar MenuBar = new JMenuBar();		
		MenuBar.setBackground(Color.black);
		window.setJMenuBar(MenuBar);		
		JMenu ProjectMenu = new JMenu("Project");
		ProjectMenu.setOpaque(true);
		//ProjectMenu.setBackground(Color.cyan);
		MenuBar.add(ProjectMenu);	
		JMenuItem OpenOption = new JMenuItem("Open...");
		ProjectMenu.add(OpenOption);
		
		JPanel MainPanel = new JPanel();
	   		MainPanel.setLayout(new BorderLayout());
	   		//MainPanel.e_add(new JLabel("Node display:",JLabel.CENTER), BorderLayout.NORTH);
	   		MainPanel.setBackground(Color.CYAN);
	   		MainPanel.add(GraphBuilder.getSurface(), BorderLayout.CENTER);

	   			JPanel controlBarPanel = new JPanel();
	   				controlBarPanel.setLayout(new FlowLayout());
	   				controlBarPanel.setComponentOrientation(ComponentOrientation.LEFT_TO_RIGHT);
	   				controlBarPanel.setBackground(Color.CYAN);
	   				randomizeNetworkButton = new JButton("RANDOMIZE EVERYTHING!");
	   				randomizeNetworkButton.addActionListener(this);
	   				controlBarPanel.add(randomizeNetworkButton);
	   				
	   				startButton = new JButton("RUN NETWORK");
	   				startButton.addActionListener(this);
	   				controlBarPanel.add(startButton);
	   				workloadInputField = new JTextField("workload");
	   				workloadInputField.addActionListener(this);
	   				workloadInputField.addMouseListener(this);
	   				workloadInputField.addMouseWheelListener(this);
	   				controlBarPanel.add(workloadInputField);   
	   				currentStepField = new JTextField("current step");
	   				currentStepField.addActionListener(this);
	   				currentStepField.addMouseListener(this);
	   				currentStepField.addMouseWheelListener(this);
	   				controlBarPanel.add(currentStepField);
	   				historyScopeField = new JTextField("history scope");
	   				historyScopeField.addActionListener(this);
	   				historyScopeField.addMouseListener(this);
	   				historyScopeField.addMouseWheelListener(this);
	   				
	   				controlBarPanel.add(historyScopeField);
	   	    MainPanel.add(controlBarPanel, BorderLayout.SOUTH);
		  
	   	   // MainPanel.addKeyListener(this);
	   	    
		window.getContentPane().add(MainPanel);  
		
		window.setVisible(true);
			
		window.addComponentListener(new ComponentListener() 
		{
		    public void componentResized(ComponentEvent e) {
		    	//change center to 0
		    //NodeGraphPanel.checkPanelScaleCenter(0, 0);//...correction nd stuff...
		    //change center to normal
		    //NodeGraphPanel.checkPanelScaleCenter(NodeGraphPanel.getWidth()/2, NodeGraphPanel.getHeight()/2);//...correction nd stuff...
		    }
			public void componentHidden(ComponentEvent arg0) {}
			public void componentMoved(ComponentEvent arg0) {}
			public void componentShown(ComponentEvent arg0) { 
				//NWPanel.update();System.out.println("updating");
			}
		});
		//NWPaintThread = new NPaintThread(NWPanel);
		//while(true) {NWPanel.update();}	
	}

	@Override
	public void actionPerformed(ActionEvent e) 
	{
		if(e.getSource()==workloadInputField) 
		{
			System.out.println("Action source: workload input field.");
			if(isInteger(workloadInputField.getText())) {
				int newWorkload = Integer.parseInt(workloadInputField.getText()); System.out.println("New workload: "+newWorkload);
				workload=newWorkload;
			}
				else
				{
					System.out.println("Workload not acceptable!");
				}
		}
		
		
		if(e.getSource()==currentStepField) {
			System.out.println("Action source: step input field.");
			if(isInteger(currentStepField.getText())) {
				int newStep = Integer.parseInt(currentStepField.getText()); System.out.println("New workload: "+newStep);
				step=newStep;
			}
				else
				{
					System.out.println("Time step not acceptable!");
				}
			
		}
		
		if(e.getSource()==historyScopeField) {
			System.out.println("Action source: history scope input field.");
			if(isInteger(historyScopeField.getText())) {
				int newScope = Integer.parseInt(historyScopeField.getText()); System.out.println("New workload: "+newScope);
				historyScope=newScope;
				GraphBuilder.setHistoryScopeTo(historyScope);
			}
				else
				{System.out.println("Time step not acceptable!");}
		}
		
		
		if(e.getSource()==startButton) {
			optainFieldInput();
			System.out.println("starting...");
			
			//while(workload>step) {
				step = GraphBuilder.runNetwork(step);
				currentStepField.setText(""+step);
			//}
				//window.addKeyListener(this);
				//window.setFocusable(true);
		}
		
		
		if(e.getSource()==randomizeNetworkButton) {
			optainFieldInput();
			System.out.println("Randomizing everything...");
			
			//while(workload>step) {
			GraphBuilder.randomizeEverything();
			//}
			
		}
		
	
	}
	
	
	private void optainFieldInput()
	{
		System.out.println("optaining workload");
		if(isInteger(workloadInputField.getText())) {
			int newWorkload = Integer.parseInt(workloadInputField.getText()); System.out.println("New workload: "+newWorkload);
			workload=newWorkload;
		}
			else
			{
				System.out.println("Workload not acceptable!");
			}
		
		System.out.println("optaining start point");
		if(isInteger(currentStepField.getText())) {
			int newStep = Integer.parseInt(currentStepField.getText()); System.out.println("New start: "+newStep);
			step=newStep;
		}
			else
			{
				System.out.println("Time step not acceptable!");
			}
		
		System.out.println("optaining history scope");
		if(isInteger(historyScopeField.getText())) {
			int newScope = Integer.parseInt(historyScopeField.getText()); System.out.println("New history scope: "+newScope);
			historyScope=newScope;
		}
			else
			{System.out.println("Time step not acceptable!");}
		
	}
	
	public static boolean isInteger(String str) 
	{
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
	        if (c < '0' || c > '9') {
	            return false;
	        }
	    }
	    return true;
	}


	@Override
	public void keyPressed(KeyEvent e) {
		
	System.out.println("Key pressed!");
		
		// TODO Auto-generated method stub
	if(e.getKeyChar()=='\n') {System.out.println("Enter!");
		if(e.getSource()==workloadInputField) {
			if(isInteger(workloadInputField.getText())) {
				int newWorkload = Integer.parseInt(workloadInputField.getText()); System.out.println("New workload: "+newWorkload);
				
				}
				else
				{
					System.out.println("Workload not acceptable!");
				}
		}
		
		
		if(e.getSource()==currentStepField) {
			if(isInteger(currentStepField.getText())) {
				int newStep = Integer.parseInt(currentStepField.getText()); System.out.println("New workload: "+newStep);
				
			}
				else
				{
					System.out.println("Time step not acceptable!");
				}
		}
	}	
	
	
	if(e.getKeyCode()==113) {GraphBuilder.getSurface().switchShadedClipping();System.out.println("shaded clipping!");}
	if(e.getKeyCode()==114) {GraphBuilder.getSurface().switchDrawRepaintSpaces();System.out.println("Repaint frames!");}
	
	if(e.getKeyCode()==115) {GraphBuilder.getSurface().switchAdvancedRendering();System.out.println("Rendering switched");}
	
	if(e.getKeyCode()==116) {GraphBuilder.getSurface().switchTouchMode();System.out.println("touch mode");}
	
	if(e.getKeyCode()==117) {GraphBuilder.getSurface().switchMapRendering();System.out.println("map rendering mode");}
    }

	@Override
	public void keyReleased(KeyEvent arg0) {
		// TODO Auto-generated method stub
		
	}


	@Override
	public void keyTyped(KeyEvent arg0) {
		// TODO Auto-generated method stub
		
	}


	@Override
	public void mouseClicked(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}


	@Override
	public void mouseEntered(MouseEvent e) {
		// TODO Auto-generated method stub
		System.out.println("entered");
		if(workloadInputField==e.getSource()) {workloadFieldEntered=true;stepFieldEntered=false;historyScopeFieldEntered=false;}

		if(currentStepField==e.getSource()) {stepFieldEntered=true;workloadFieldEntered=false;historyScopeFieldEntered=false;}

		if(historyScopeField==e.getSource()) {historyScopeFieldEntered=true;stepFieldEntered=false;workloadFieldEntered=false;}
	}


	@Override
	public void mouseExited(MouseEvent e) {
		// TODO Auto-generated method stub
		if(workloadInputField==e.getSource()) {workloadFieldEntered=false;stepFieldEntered=false;historyScopeFieldEntered=false;}

		if(currentStepField==e.getSource()) {stepFieldEntered=false;workloadFieldEntered=false;historyScopeFieldEntered=false;}

		if(historyScopeField==e.getSource()) {historyScopeFieldEntered=false;stepFieldEntered=false;workloadFieldEntered=false;}
	}


	@Override
	public void mousePressed(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}


	@Override
	public void mouseReleased(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}


	@Override
	public void mouseWheelMoved(MouseWheelEvent e) 
	{
		// TODO Auto-generated method stub
		
		System.out.println("mouse wheel moved");
		
		if(workloadFieldEntered) 
		   {System.out.println("optaining workload");
		    if(isInteger(workloadInputField.getText())) 
		       { int newWorkload = Integer.parseInt(workloadInputField.getText()); System.out.println("New workload: "+newWorkload);
			    workload=newWorkload;
		       }
			    else
			   {System.out.println("Workload not acceptable!");
			    workload=0;
			   }
		
		    workload+=e.getWheelRotation();
		    workloadInputField.setText(""+workload);
		   }

		if(stepFieldEntered) 
		   {
			System.out.println("optaining step");
		    if(isInteger(currentStepField.getText())) 
		       {int newStep = Integer.parseInt(currentStepField.getText()); System.out.println("New step: "+newStep);
			    step=newStep;
		       }
			   else
			   {
				System.out.println("Time step not acceptable!");
				step = 0;
			   }
		    step+=e.getWheelRotation();
		    currentStepField.setText(""+step);
		   }

		if(historyScopeFieldEntered) 
		   {
			System.out.println("optaining history scope");
		    if(isInteger(historyScopeField.getText())) 
		       {
			    int newScope = Integer.parseInt(historyScopeField.getText()); System.out.println("New history scope: "+newScope);
			    historyScope=newScope;
		       }
			else
			   {System.out.println("History scope not acceptable!");
			   
			   }
		     historyScope+=e.getWheelRotation();
		     historyScopeField.setText(""+historyScope);
		    }
	}

	
	
}
