
package neureka.utility;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Font;

import javax.swing.JFrame;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.text.BadLocationException;


public class NMessageFrame extends JFrame{
	
	JTextArea outputField;
	JTextField eingabefeld;
	boolean entered;
	String input;
	int linecount;
	//=================================================
	public NMessageFrame(String title, int maxLineCount) 
	{
		construct(title, maxLineCount);
	}
	public NMessageFrame(String title) 
	{
		construct(title, 1000);
	}
	private void construct(String title, int maxLineCount)
	{
		linecount = maxLineCount;
		input = "";
		entered = false;
		this.setTitle(title);
		this.setSize(400, 400);
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		this.setLocationRelativeTo(null);

		outputField = new JTextArea();
		outputField.setText("Process messages:\n");
		outputField.setEditable(false);
		
		Font txt = new Font("Verdana", Font.LAYOUT_LEFT_TO_RIGHT, 15);
		outputField.setBackground(Color.BLACK);
		outputField.setForeground(Color.CYAN);
		outputField.setFont(txt);

		JScrollPane scroll = new JScrollPane(outputField);
		scroll.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_ALWAYS);
		scroll.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_ALWAYS);
		scroll.setBackground(Color.DARK_GRAY);
		scroll.setForeground(Color.BLUE);
		scroll.getVerticalScrollBar().setBackground(Color.DARK_GRAY);
		scroll.getHorizontalScrollBar().setBackground(Color.DARK_GRAY);
		
		this.getContentPane().add(scroll, BorderLayout.CENTER);
		this.setVisible(true);
	}
	
	public void println(String text) 
	{
		outputField.setText(outputField.getText() + text + "\n");
		if (outputField.getLineCount() > 1000) 
		{
			try 
			{
				outputField.replaceRange("", 0, outputField.getLineEndOffset(0));
			} 
			catch (BadLocationException e) 
			{
				e.printStackTrace();
			}
		}
	}
	public void print(String text) 
	{
		outputField.setText(outputField.getText() + text);
		if (outputField.getLineCount() > 1000) 
		{
			try 
			{
				outputField.replaceRange("", 0, outputField.getLineEndOffset(0));
			} 
			catch (BadLocationException e) 
			{
				e.printStackTrace();
			}
		}
	}
	public String read() 
	{
		entered = false;
		while (!entered) 
		{
			try 
			{
				Thread.sleep(10);
			} 
			catch (InterruptedException e) 
			{
				e.printStackTrace();
			}
		}
		return input;
	}
}
