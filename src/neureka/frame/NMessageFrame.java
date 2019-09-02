
package neureka.frame;

import java.awt.*;

import javax.swing.*;
import javax.swing.text.BadLocationException;


public class NMessageFrame extends JFrame
{
	JScrollPane scroll;
	private static int windowCount = 0;
	private static int width = 500;
	private static int height = 400;
	private static int wpos = 0;
	private static int hpos = 0;
	private JTextArea outputField;
	private boolean entered;
	private String input;
	private int linecount;
	//=================================================
	public NMessageFrame(String title) 
	{
		construct(title, 1000);
	}
	public NMessageFrame(String title, int maxLineCount)
	{
		construct(title, maxLineCount);
	}
	private void construct(String title, int maxLineCount)
	{
		Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
		int wdiv = 3;
		int hdiv = 2;
		this.width = (int)screenSize.getWidth()/wdiv;
		this.height = (int)(screenSize.getHeight()*0.925)/hdiv;
		this.wpos = windowCount%(wdiv);
		this.hpos = windowCount/(wdiv);
		windowCount++;
		linecount = maxLineCount;
		input = "";
		entered = false;
		this.setTitle(title);
		this.setSize(this.width, this.height);
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		this.setLocation(wpos*width, hpos*height);

		outputField = new JTextArea();
		outputField.setText("Process messages:\n");
		outputField.setEditable(false);
		
		Font txt = new Font("Verdana", Font.LAYOUT_LEFT_TO_RIGHT, 15);
		outputField.setBackground(Color.BLACK);
		outputField.setForeground(Color.CYAN);
		outputField.setFont(txt);

		scroll = new JScrollPane(outputField);
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
				scroll.getViewport().setViewPosition(new java.awt.Point(0, scroll.getViewport().getHeight()));
			} 
			catch (BadLocationException e) 
			{
				e.printStackTrace();
			}
		}
	}

	public void bottom(){
		JScrollBar vertical = scroll.getVerticalScrollBar();
		vertical.setValue( vertical.getMaximum()+10 );
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
