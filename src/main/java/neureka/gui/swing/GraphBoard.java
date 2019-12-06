package neureka.gui.swing;

import neureka.Tsr;
import neureka.function.factory.autograd.GraphNode;

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


public class GraphBoard implements ActionListener, KeyListener, MouseListener, MouseWheelListener {
    private static JFrame _window;

    public static final int WIDTH = 800;
    public static final int HEIGHT = 700;

    GraphSurfaceBuilder GraphBuilder;

    private JButton startButton;
    private JButton randomizeNetworkButton;

    public GraphBoard(Tsr source) {
        //fullscreen
		/*
			frame.setExtendedState(JFrame.MAXIMIZED_BOTH);
			frame.setUndecorated(true);
			frame.setVisible(true);
		*/
        GraphBuilder = new GraphSurfaceBuilder((GraphNode) source.find(GraphNode.class));

        System.setProperty("sun.java2d.opengl", "true");
        _window = new JFrame("Network Display");
        _window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        _window.setSize(WIDTH, HEIGHT);
        _window.setLocationRelativeTo(null);
        _window.setResizable(true);
        //_window.setLayout(null);
        _window.addKeyListener(this);
        _window.setFocusable(true);
        _window.setUndecorated(false);
        //_window.getRootPane().setWindowDecorationStyle(JRootPane.BOTTOM_ALIGNMENT);
        //((JComponent) _window).setBorder(new EmptyBorder(5, 5, 5, 5));
        // NodeGraphPanel.addKeyListener(this);
        // NodeGraphPanel.setFocusable(true);

        //_window.setExtendedState(JFrame.MAXIMIZED_BOTH);
        //_window.setUndecorated(true);
        _window.setVisible(true);

        JMenuBar MenuBar = new JMenuBar();
        MenuBar.setBackground(Color.black);
        _window.setJMenuBar(MenuBar);
        JMenu ProjectMenu = new JMenu("Project");
        ProjectMenu.setOpaque(true);
        //ProjectMenu.setBackground(Color.cyan);
        MenuBar.add(ProjectMenu);
        JMenuItem OpenOption = new JMenuItem("Open...");
        ProjectMenu.add(OpenOption);

        JPanel MainPanel = new JPanel();
        MainPanel.setLayout(new BorderLayout());
        //MainPanel.addInto(new JLabel("AbstractSurfaceNode display:",JLabel.CENTER), BorderLayout.NORTH);
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
        MainPanel.add(controlBarPanel, BorderLayout.SOUTH);

        _window.getContentPane().add(MainPanel);
        _window.setVisible(true);

        _window.addComponentListener(new ComponentListener() {
            public void componentResized(ComponentEvent e) {
                //change center to 0
                //NodeGraphPanel.checkPanelScaleCenter(0, 0);//...correction nd stuff...
                //change center to normal
                //NodeGraphPanel.checkPanelScaleCenter(NodeGraphPanel.getWidth()/2, NodeGraphPanel.getHeight()/2);//...correction nd stuff...
            }

            public void componentHidden(ComponentEvent arg0) {
            }

            public void componentMoved(ComponentEvent arg0) {
            }

            public void componentShown(ComponentEvent arg0) {
            }
        });
    }

    public GraphSurfaceBuilder getBuilder(){
        return GraphBuilder;
    }

    @Override
    public void actionPerformed(ActionEvent e) {

        if (e.getSource() == startButton) {
            optainFieldInput();

            //while(workload>step) {
            //step = GraphBuilder.runNetwork(step);
            //_currentStepField.setText(""+step);
            //}
            //_window.addKeyListener(this);
            //_window.setFocusable(true);
        }


        if (e.getSource() == randomizeNetworkButton) {
            optainFieldInput();

            //while(workload>step) {
            //GraphBuilder.randomizeEverything();
            //}

        }


    }


    private void optainFieldInput() {

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
            if (c < '0' || c > '9') {
                return false;
            }
        }
        return true;
    }


    @Override
    public void keyPressed(KeyEvent e) {

        if (e.getKeyCode() == 113) {
            GraphBuilder.getSurface().switchShadedClipping();
            System.out.println("shaded clipping!");
        }
        if (e.getKeyCode() == 114) {
            GraphBuilder.getSurface().switchDrawRepaintSpaces();
            System.out.println("Repaint frames!");
        }

        if (e.getKeyCode() == 115) {
            GraphBuilder.getSurface().switchAdvancedRendering();
            System.out.println("Rendering switched");
        }

        if (e.getKeyCode() == 116) {
            GraphBuilder.getSurface().switchTouchMode();
            System.out.println("neureka.ngui.touch mode");
        }

        if (e.getKeyCode() == 117) {
            GraphBuilder.getSurface().switchMapRendering();
            System.out.println("map rendering mode");
        }
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
    }


    @Override
    public void mouseExited(MouseEvent e) {
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
    public void mouseWheelMoved(MouseWheelEvent e) {
    }


}
