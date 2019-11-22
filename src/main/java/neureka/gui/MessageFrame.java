
package neureka.gui;

import java.awt.*;
import java.awt.event.WindowEvent;

import javax.swing.*;
import javax.swing.text.BadLocationException;


public class MessageFrame extends JFrame
{
	JScrollPane _scroll;
	private static int _window_count = 0;
	private static int _width = 500;
	private static int _height = 400;
	private static int _wpos = 0;
	private static int _hpos = 0;
	private JTextArea _output_field;
	private int _linecount;
	//=================================================
	public MessageFrame(String title)
	{
		construct(title, 1000);
	}
	public MessageFrame(String title, int maxLineCount)
	{
		construct(title, maxLineCount);
	}
	private void construct(String title, int maxLineCount)
	{
		Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
		int wdiv = 3;
		int hdiv = 3;
		_width = (int)screenSize.getWidth()/wdiv;
		_height = (int)(screenSize.getHeight()*0.925)/hdiv;
		_wpos = _window_count %(wdiv);
		_hpos = _window_count /(wdiv);
		_window_count++;
		_linecount = maxLineCount;
		this.setTitle(title);
		this.setSize(this._width, this._height);
		this.setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);
		this.setLocation(_wpos * _width, _hpos * _height);

		_output_field = new JTextArea();
		_output_field.setText("Process messages:\n");
		_output_field.setEditable(false);
		
		Font txt = new Font("Verdana", Font.LAYOUT_LEFT_TO_RIGHT, 15);
		_output_field.setBackground(Color.BLACK);
		_output_field.setForeground(Color.CYAN);
		_output_field.setFont(txt);

		_scroll = new JScrollPane(_output_field);
		_scroll.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_ALWAYS);
		_scroll.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_ALWAYS);
		_scroll.setBackground(Color.DARK_GRAY);
		_scroll.setForeground(Color.BLUE);
		_scroll.getVerticalScrollBar().setBackground(Color.DARK_GRAY);
		_scroll.getHorizontalScrollBar().setBackground(Color.DARK_GRAY);
		this.getContentPane().add(_scroll, BorderLayout.CENTER);
		this.setVisible(true);
	}
	
	public void println(String text) 
	{
		_output_field.setText(_output_field.getText() + text + "\n");
		if (_output_field.getLineCount() > _linecount)
		{
			try 
			{
				_output_field.replaceRange("", 0, _output_field.getLineEndOffset(0));
			}
			catch (BadLocationException e) 
			{
				e.printStackTrace();
			}
		}
	}
	public void print(String text) 
	{
		_output_field.setText(_output_field.getText() + text);
		if (_output_field.getLineCount() > _linecount)
		{
			try 
			{
				_output_field.replaceRange("", 0, _output_field.getLineEndOffset(0));
				_scroll.getViewport().setViewPosition(new java.awt.Point(0, _scroll.getViewport().getHeight()));
			} 
			catch (BadLocationException e) 
			{
				e.printStackTrace();
			}
		}
	}

	public void bottom(){
		JScrollBar vertical = _scroll.getVerticalScrollBar();
		vertical.setValue( vertical.getMaximum()+10 );
	}

	public void close(){
		this.dispatchEvent(new WindowEvent(this, WindowEvent.WINDOW_CLOSING));
	}

}
