
package neureka.gui.swing;

public class SurfaceRepaintSpace {

    private double _TLX, _TLY, _BRX, _BRY;

    public SurfaceRepaintSpace(
            double TLX,
            double TLY,
            double BRX,
            double BRY
    ) {
        _TLX = TLX;
        _TLY = TLY;
        _BRX = BRX;
        _BRY = BRY;

    }

    public boolean contains(SurfaceRepaintSpace other){
        if(_TLX<=other._TLX && _TLY<=other._TLY && _BRX>=other._BRX && _BRY>=other._BRY){
            return true;
        }
        return false;
    }

    public double getWidth(){
        return _BRX-_TLX;
    }
    public double getHeight(){
        return _BRY-_TLY;
    }

    public double getLeftPeripheral(){
        return _TLX;
    }

    public double getTopPeripheral(){
        return _TLY;
    }

    public double getRightPeripheral(){
        return _BRX;
    }

    public double getBottomPeripheral(){
        return _BRY;
    }


}
