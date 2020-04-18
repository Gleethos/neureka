package neureka;

import groovy.lang.Closure;
import groovy.lang.GroovyShell;
import groovy.lang.GroovySystem;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class Neureka
{
    private static final Map<Thread, Neureka> _instances;
    private static String _settings_source;
    private static String _setup_source;

    private Settings _settings;
    private Utility _utility;

    static {
        _instances = new ConcurrentHashMap<>();
    }

    private Neureka(){
        _settings = new Settings();
        _utility = new Utility();
    }

    public static Neureka instance(){
        return instance(Thread.currentThread());
    }

    public static void setContext(Thread thread, Neureka instance){
        _instances.put(thread, instance);
    }

    public static Neureka instance(Thread thread){
        if(_instances.containsKey(thread)) return _instances.get(thread);
        else {
            Neureka instance = new Neureka();
            setContext(thread, instance);
            synchronized (Neureka.class) {
                instance.reset();
            }
            return instance;
        }
    }

    public static Neureka instance(Closure c) {
        c.setDelegate(Neureka.instance());
        c.call();
        return Neureka.instance();
    }

    public Settings settings(){
        return _settings;
    }

    public Settings settings(Closure c){
        c.setDelegate(_settings);
        c.call();
        return _settings;
    }

    public Utility utility(){
        return _utility;
    }

    public static String version(){
        return "1.0.0";
    }

    public void reset() {
        if (_settings_source == null || _setup_source == null) {
            _settings_source = utility().readResource("library_settings.groovy");
            _setup_source = utility().readResource("scripting_setup.groovy");
        }
        try {
            String version = GroovySystem.getVersion();
            if(Integer.parseInt(version.split("\\.")[0]) < 3) {
                throw new IllegalCallerException(
                        "Wrong groovy version "+version+" found! Version 3.0.0 or greater required."
                );
            }
            new GroovyShell(this.getClass().getClassLoader()).evaluate(_settings_source);
            new GroovyShell(this.getClass().getClassLoader()).evaluate(_setup_source);
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    private boolean _currentThreadIsAuthorized(){
        return this.equals(_instances.get(Thread.currentThread()));
    }

    public class Settings
    {
        private Debug _debug;
        private AutoDiff _autoDiff;
        private Indexing _indexing;
        private View _view;

        private boolean _isLocked = false;

        private Settings() {
            _debug = new Debug();
            _autoDiff = new AutoDiff();
            _indexing = new Indexing();
            _view = new View();
        }

        public Debug debug() {
            return _debug;
        }

        public Debug debug(Closure c) {
            c.setDelegate(_debug);
            c.call();
            return _debug;
        }

        public AutoDiff autoDiff(){
            return _autoDiff;
        }

        public AutoDiff autoDiff(Closure c) {
            c.setDelegate(_autoDiff);
            c.call();
            return _autoDiff;
        }

        public Indexing indexing(){
            return _indexing;
        }

        public Indexing indexing(Closure c) {
            c.setDelegate(_indexing);
            c.call();
            return _indexing;
        }

        public View view(){
            return _view;
        }

        public View view(Closure c) {
            c.setDelegate(_view);
            c.call();
            return _view;
        }

        public boolean isLocked(){
            return  _isLocked;
        }

        public void setIsLocked(boolean locked) {
            _isLocked = locked;
        }

        public class Debug
        {
            /**
             * Every derivative is calculated with respect to some graph node.
             * Graph nodes contain payload tensors.
             * A tensor might not always be used for backpropagation.
             * Therefore it will be deleted if possible.
             * Targeted tensors are either leave tensors (They require gradients)
             * or they are angle points between forward- and reverse-mode-AutoDiff!
             * In this case:
             * If the tensor is not needed for backpropagation it will be deleted.
             * The graph node will dereference the tensor either way.
             *
             * The flag determines this behavior with respect to target nodes.
             * It is used in the test suit to validate that the right tensors were calculated.
             * This flag should not be modified in production! (memory leak)
             */
            private boolean _keepDerivativeTargetPayloads;

            public boolean keepDerivativeTargetPayloads(){
                return _keepDerivativeTargetPayloads;
            }

            public void setKeepDerivativeTargetPayloads(boolean keep){
                if(_isLocked || !_currentThreadIsAuthorized()) return;
                _keepDerivativeTargetPayloads = keep;
            }

        }

        public class AutoDiff // Auto-Differentiation
        {
            /**
             * This flag enables an optimization technique which only applies
             * gradients as soon as they are needed by a tensor (the tensor is used again).
             * If the flag is set to true
             * then error values will accumulate whenever it makes sense.
             * This technique however uses more memory but will
             * improve performance for some networks substantially.
             * The technique is termed JIT-Propagation.
             */
            private boolean _retainPendingErrorForJITProp;

            /**
             * Gradients will automatically be applied to tensors as soon as
             * they are being used for calculation.
             * This feature works well with JIT-Propagation.
             */
            private boolean _applyGradientWhenTensorIsUsed;

            public boolean retainPendingErrorForJITProp(){
                return _retainPendingErrorForJITProp;
            }

            public void setRetainPendingErrorForJITProp(boolean retain){
                if(_isLocked || !_currentThreadIsAuthorized()) return;
                _retainPendingErrorForJITProp = retain;
            }

            public boolean applyGradientWhenTensorIsUsed(){
                return _applyGradientWhenTensorIsUsed;
            }

            public void setApplyGradientWhenTensorIsUsed(boolean apply){
                if(_isLocked || !_currentThreadIsAuthorized()) return;
                _applyGradientWhenTensorIsUsed = apply;
            }

        }

        public class Indexing
        {
            private boolean _legacyIndexing;

            private boolean _thoroughIndexing;

            public boolean legacy(){
                return _legacyIndexing;
            }

            public void setLegacy(boolean enabled){
                if(_isLocked || !_currentThreadIsAuthorized()) return;
                _legacyIndexing = enabled;//NOTE: gpu code must recompiled! (in OpenCLPlatform)
            }

            public boolean thorough(){
                return _thoroughIndexing;
            }

            public void setThorough(boolean thorough){
                if(_isLocked || !_currentThreadIsAuthorized()) return;
                _thoroughIndexing = thorough;
            }

        }

        public class View
        {

            private boolean _legacyView;

            public boolean legacy(){
                return _legacyView;
            }

            public void setLegacy(boolean enabled){
                if(_isLocked || !_currentThreadIsAuthorized()) return;
                _legacyView = enabled;
            }

        }

    }

    public static class Utility {
        /**
         * Helper method which reads the file with the given name and returns
         * the contents of this file as a String. Will exit the application
         * if the file can not be read.
         *
         * @param path
         * @return The contents of the file
         */
        public String readResource(String path){
            InputStream stream = getClass().getClassLoader().getResourceAsStream(path);
            try {
                BufferedReader br = new BufferedReader(new InputStreamReader(stream));//new FileInputStream(fileName)));
                StringBuffer sb = new StringBuffer();
                String line = "";
                while (line!=null) {
                    line = br.readLine();
                    if (line != null) sb.append(line).append("\n");
                }
                return sb.toString();
            } catch (IOException e) {
                e.printStackTrace();
                System.exit(1);
                return null;
            }
        }





    }


}
