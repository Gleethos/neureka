package neureka.devices.file.handles.util;

import neureka.Tsr;
import neureka.devices.file.FileHandle;
import neureka.devices.file.handles.CSVHandle;
import neureka.devices.file.handles.IDXHandle;
import neureka.devices.file.handles.JPEGHandle;

import java.util.HashMap;
import java.util.Map;

/**
 *  This class is a simple wrapper around "Loader" and "Saver" lambdas
 *  which instantiate {@link FileHandle} classes.
 *  <b>This is an internal class. Do not depend on it!</b>
 */
public final class HandleFactory
{
    public interface Loader
    {
        FileHandle load(String name, Map<String, Object> config);
    }

    public interface Saver
    {
        FileHandle save(String name, Tsr tensor, Map<String, Object> config);
    }

    private final Map<String, Loader> _LOADERS = new HashMap<>();
    private final Map<String, Saver> _SAVERS = new HashMap<>();

    public HandleFactory() {
        _LOADERS.put("idx", (name, conf) -> new IDXHandle(name));
        _LOADERS.put("jpg", (name, conf) -> new JPEGHandle(name));
        _LOADERS.put("png", (name, conf) -> null); // TODO!
        _LOADERS.put("csv", (name, conf) -> new CSVHandle(name, conf));

        _SAVERS.put("idx", (name, tensor, conf) -> new IDXHandle(tensor, name));
        _SAVERS.put("jpg", (name, tensor, conf) -> new JPEGHandle(tensor, name));
        _SAVERS.put("png", (name, tensor, conf) -> null); // TODO!
        _SAVERS.put("csv", (name, tensor, conf) -> new CSVHandle(tensor, name));
    }

    public boolean hasLoader(String name){
        return _LOADERS.containsKey(name);
    }

    public boolean hasSaver(String name){
        return _SAVERS.containsKey(name);
    }

    public Loader getLoader(String name) {
        return _LOADERS.get(name);
    }

    public Saver getSaver(String name) {
        return _SAVERS.get(name);
    }
}