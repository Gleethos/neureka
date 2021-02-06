package neureka.devices.file.heads.util;

import neureka.Tsr;
import neureka.devices.file.FileHead;
import neureka.devices.file.heads.CSVHead;
import neureka.devices.file.heads.IDXHead;
import neureka.devices.file.heads.JPEGHead;

import java.util.HashMap;
import java.util.Map;

public class HeadFactory
{
    public interface Loader {
        FileHead load(String name, Map<String, Object> config);
    }

    public interface Saver {
        FileHead save(String name, Tsr tensor, Map<String, Object> config);
    }

    private final Map<String, Loader> _LOADERS = new HashMap<>();
    private final Map<String, Saver> _SAVERS = new HashMap<>();

    public HeadFactory() {
        _LOADERS.put("idx", (name, conf) -> new IDXHead(name));
        _LOADERS.put("jpg", (name, conf) -> new JPEGHead(name));
        _LOADERS.put("png", (name, conf) -> null); // TODO!
        _LOADERS.put("csv", (name, conf) -> new CSVHead(name, conf));

        _SAVERS.put("idx", (name, tensor, conf) -> new IDXHead(tensor, name));
        _SAVERS.put("jpg", (name, tensor, conf) -> new JPEGHead(tensor, name));
        _SAVERS.put("png", (name, tensor, conf) -> null); // TODO!
        _SAVERS.put("csv", (name, tensor, conf) -> new CSVHead(tensor, name));
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