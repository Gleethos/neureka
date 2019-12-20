package neureka.gui.swing;

import java.awt.Graphics2D;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;

public abstract class AbstractSpaceMap {

	static public int MAX = 8;
	
	public abstract AbstractSpaceMap addAll(LinkedList<SurfaceObject> elements);
	
	public abstract AbstractSpaceMap addAndUpdate(SurfaceObject node);
	
	public abstract AbstractSpaceMap removeAndUpdate(SurfaceObject node);
	
	public abstract int getCount();

	public abstract LinkedList<SurfaceObject> getAll();
	public abstract LinkedList<SurfaceObject> getAllWithin(double[] frame);

	public abstract LinkedList<SurfaceObject> findAllAt(double x, double y, MapAction Actor);
	public abstract LinkedList<SurfaceObject> findAllWithin(double[] frame, MapAction Actor);

	public abstract boolean applyToAll(MapAction Actor);
	public abstract boolean applyToAllWithin(double[] frame, MapAction Actor);
	
	public abstract void paintStructure(Graphics2D brush);

	public interface MapAction         {boolean act(SurfaceObject thing);}

	protected static List<SurfaceObject> _withinFrame(LinkedList<SurfaceObject> list, HashMap<SurfaceObject, SurfaceObject> elements, double[] frame){
		if(elements!=null){
			elements.forEach(
					(k, o)->
					{
						double LP = frame[0]; double RP = frame[1];
						double TP = frame[2]; double BP = frame[3];
						if(
								o.getRightPeripheral() < RP && o.getLeftPeripheral() > LP &&
								o.getBottomPeripheral() < BP && o.getTopPeripheral() > TP
						) {
							list.add(o);
						}
					});
		}
		return list;
	}

}
