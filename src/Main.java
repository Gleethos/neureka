import java.awt.Component;
import java.awt.Container;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
/* ww  w.jav  a 2s  .  co m*/
import javax.swing.AbstractButton;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class Main {
  public void buildGUI() {
    JFrame.setDefaultLookAndFeelDecorated(true);
    JFrame f = new JFrame();
    f.setResizable(true);
    removeMinMaxClose(f);
    JPanel p = new JPanel(new GridBagLayout());
    JButton btn = new JButton("Exit");
    p.add(btn, new GridBagConstraints());
    f.getContentPane().add(p);
    f.setSize(400, 300);
    f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    f.setVisible(true);
    btn.addActionListener(e -> System.exit(0));
  }
  public void removeMinMaxClose(Component comp) {
    if (comp instanceof AbstractButton) {
      comp.getParent().remove(comp);
    }
    if (comp instanceof Container) {
      Component[] comps = ((Container) comp).getComponents();
      for (int x = 0, y = comps.length; x < y; x++) {
        removeMinMaxClose(comps[x]);
      }
    }
  }
  public static void main(String[] args) {
    new Main().buildGUI();
  }
}
