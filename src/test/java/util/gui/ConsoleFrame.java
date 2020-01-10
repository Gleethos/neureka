package util.gui;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.AdjustmentListener;
import java.awt.event.AdjustmentEvent;

import javax.swing.JFrame;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.text.BadLocationException;

public class ConsoleFrame extends JFrame{

	private JTextArea _outputField;
	private JTextField _inputField;
	private boolean _receptive = true;
	private String _input = "";
	private int _linecount;
	private boolean _entered = false;

	public ConsoleFrame(String title, int maxLineCount) {
			_linecount = maxLineCount;
			_input = "";
			_entered = false;
			this.setTitle(title);
			this.setSize(800, 400);
			this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
			this.setLocationRelativeTo(null);

		_outputField = new JTextArea();
		_outputField.setText("This little area here is where you getFrom your feedback:  :D " + "\n");
		// _outputField.setBorder(BorderFactory.createLineBorder(Color.black));
		_outputField.setEditable(false);

		Font txt = new Font("Verdana", Font.LAYOUT_LEFT_TO_RIGHT, 15);
		_outputField.setBackground(Color.BLACK);
		_outputField.setForeground(Color.CYAN);
		_outputField.setFont(txt);

		JScrollPane scroll = new JScrollPane(_outputField);
		scroll.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_ALWAYS);
		scroll.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_ALWAYS);
		scroll.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_ALWAYS);
		scroll.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_ALWAYS);
		scroll.setBackground(Color.DARK_GRAY);
		scroll.setForeground(Color.BLUE);
		scroll.getVerticalScrollBar().setBackground(Color.DARK_GRAY);
		scroll.getHorizontalScrollBar().setBackground(Color.DARK_GRAY);

		 scroll.getVerticalScrollBar().addAdjustmentListener(
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
				});
		this.add(scroll, BorderLayout.CENTER);

		// this.addInto(_outputField, BorderLayout.CENTER);

		_inputField = new JTextField();
		_inputField.setToolTipText("Insert _input here!");
		_inputField.setPreferredSize(new Dimension(0, 40));
		_inputField.setBackground(Color.BLACK);
		_inputField.setForeground(Color.CYAN);
		_inputField.setFont(txt);
		this.add(_inputField, BorderLayout.PAGE_END);

		// Eingabefeld Enter-Listener
		_inputField.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				if (_receptive) {
					_input = _inputField.getText();
					_entered = true;
					_inputField.setText("");
				}
			}
		});
		this.getContentPane().add(scroll, BorderLayout.CENTER);
		this.setVisible(true);
	}

	public void println(String text) {
		_outputField.setText(_outputField.getText() + text + "\n");
		if (_outputField.getLineCount() > _linecount) {
			try {
				_outputField.replaceRange("", 0, _outputField.getLineEndOffset(0));
			} catch (BadLocationException e) {
				e.printStackTrace();
			}
		}
	}

	public void print(String text) {
		_outputField.setText(_outputField.getText() + text);
		if (_outputField.getLineCount() > _linecount) {
			try {
				_outputField
						.replaceRange("", 0, _outputField.getLineEndOffset(0));
			} catch (BadLocationException e) {
				e.printStackTrace();
			}
		}
	}

	public String read() {
		_entered = false;
		while (!_entered) {
			try {
				Thread.sleep(10);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		return _input;
	}
	
	
	
	
}
