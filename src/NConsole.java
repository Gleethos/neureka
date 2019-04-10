
import napi.main.core.NVNode;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JFrame;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.text.BadLocationException;


public class NConsole extends JFrame{
	
	JTextArea ausgabefeld;
	JTextField eingabefeld;
   
	boolean entered = false;
	boolean check = false;
	String input = "";
    
	public NConsole(String title) {

		this.setTitle(title);
		this.setSize(800, 400);
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		this.setLocationRelativeTo(null);

		ausgabefeld = new JTextArea();
		ausgabefeld.setText("This little area here is where you get your feedback:  :D " + "\n");
		// ausgabefeld.setBorder(BorderFactory.createLineBorder(Color.black));
		ausgabefeld.setEditable(false);
	

		JScrollPane scroll = new JScrollPane(ausgabefeld);
		scroll.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_ALWAYS);
		scroll.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_ALWAYS);
		/* scroll.getVerticalScrollBar().addAdjustmentListener(
				new AdjustmentListener() {
					public void adjustmentValueChanged(AdjustmentEvent e) {
						e.getAdjustable().setValue(
								e.getAdjustable().getMaximum());
					}
				});
		scroll.getHorizontalScrollBar().addAdjustmentListener(
				new AdjustmentListener() {
					public void adjustmentValueChanged(AdjustmentEvent e) {
						e.getAdjustable().setValue(
								e.getAdjustable().getMaximum());
					}
				});*/
		this.add(scroll, BorderLayout.CENTER);

		// this.add(ausgabefeld, BorderLayout.CENTER);

		eingabefeld = new JTextField();
		eingabefeld.setToolTipText("Insert input here!");
		eingabefeld.setPreferredSize(new Dimension(0, 40));
		this.add(eingabefeld, BorderLayout.PAGE_END);

		// Eingabefeld Enter-Listener
		eingabefeld.addActionListener(new ActionListener() {

			public void actionPerformed(ActionEvent e) {
				if (check) {
					input = eingabefeld.getText();
					entered = true;
					eingabefeld.setText("");
				}
			}
		});

		this.setVisible(true);

	}

	public void addConsoleNeuron(NVNode Neuron){println("ThreadNeuron object added! - Console now usable!");}

	public void println(String text) {
		ausgabefeld.setText(ausgabefeld.getText() + text + "\n");
		if (ausgabefeld.getLineCount() > 1000) {
			try {
				ausgabefeld.replaceRange("", 0, ausgabefeld.getLineEndOffset(0));
			} catch (BadLocationException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

	
	public void print(String text) {
		ausgabefeld.setText(ausgabefeld.getText() + text);
		if (ausgabefeld.getLineCount() > 1000) {
			try {
				ausgabefeld
						.replaceRange("", 0, ausgabefeld.getLineEndOffset(0));
			} catch (BadLocationException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
	

	public void setCheck(boolean b) {
		check = b;
	}

	public String read() {
		entered = false;

		while (!entered) {
			try {
				Thread.sleep(10);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		return input;
	}
	
	
	
	
}
