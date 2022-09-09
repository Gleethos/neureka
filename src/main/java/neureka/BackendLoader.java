package neureka;

import neureka.backend.api.*;
import neureka.backend.api.annotations.Backend;
import neureka.backend.api.annotations.Loadable;
import neureka.devices.Device;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

public class BackendLoader
{
    private static final Logger _LOG = LoggerFactory.getLogger(BackendLoader.class);

    public static void findAndLoad(BackendContext context, String... packages) {
        for ( String pkg : packages ) {
            ImplementationPackage implementationPackage = null;
            try {
                implementationPackage = _find(pkg);
            } catch (IOException | ClassNotFoundException e) {
                _LOG.error("Could not load backend implementations from package: " + pkg, e);
            }
            if ( implementationPackage != null ) {
                _LOG.debug("Loading backend implementations from package: " + pkg);
                implementationPackage.load(context);
            }
        }
    }

    private static ImplementationPackage _find(String packageName ) throws ClassNotFoundException, IOException
    {
        Package pkg = Class.forName(packageName + ".package-info").getPackage();
        Backend backendInfo = pkg.getAnnotation(Backend.class);
        /*
            Now we load all the classes in the package and check if they have the @Loadable annotation.
         */
        List<Class> classes = getClassesForPackage(pkg);
        List<LoadableImplementation> implementations = new ArrayList<>();
        for (Class packageClass : classes) {
            Loadable loadable = (Loadable) packageClass.getAnnotation(Loadable.class);
            if (loadable != null) {
                if ( !ImplementationFor.class.isAssignableFrom(packageClass) )
                    throw new RuntimeException("The @Loadable annotation can only be used on classes that implement the interface '"+ImplementationFor.class+"'!");

                implementations.add(new LoadableImplementation(
                    packageClass, loadable.operation(), loadable.algorithm(), backendInfo.device()
                ));
            }
        }
        return new ImplementationPackage(
                    pkg.getName(),
                    backendInfo.device(),
                    implementations
                );
    }

    private static List<Class> getClassesForPackage(Package pkg)
    {
        String pkgname = pkg.getName();

        List<Class> classes = new ArrayList<>();

        // Get a File object for the package
        File directory = null;
        String fullPath;
        String relPath = pkgname.replace('.', '/');
        URL resource = ClassLoader.getSystemClassLoader().getResource(relPath);
        if ( resource == null )
            throw new RuntimeException("No resource for " + relPath);

        fullPath = resource.getFile();
        try {
            directory = new File(resource.toURI());
        } catch (URISyntaxException e) {
            throw new RuntimeException(pkgname + " (" + resource + ") does not appear to be a valid URL / URI.  Strange, since we got it from the system...", e);
        } catch (IllegalArgumentException e) {
            directory = null;
        }

        if ( directory != null && directory.exists() ) {

            // Get the list of the files contained in the package
            String[] files = directory.list();
            for ( String file : files ) {

                // we are only interested in .class files
                if ( file.endsWith(".class") ) {

                    // removes the .class extension
                    String className = pkgname + '.' + file.substring(0, file.length() - 6);
                    try {
                        classes.add(Class.forName(className));
                    } catch (ClassNotFoundException e) {
                        throw new RuntimeException("ClassNotFoundException loading " + className);
                    }
                }
            }
        } else {
            try {
                String jarPath = fullPath.replaceFirst("[.]jar[!].*", ".jar").replaceFirst("file:", "");
                JarFile jarFile = new JarFile(jarPath);
                Enumeration<JarEntry> entries = jarFile.entries();
                while (entries.hasMoreElements()) {
                    JarEntry entry = entries.nextElement();
                    String entryName = entry.getName();
                    if ( entryName.startsWith(relPath) && entryName.length() > (relPath.length() + "/".length()) ) {
                        String className = entryName.replace('/', '.').replace('\\', '.').replace(".class", "");
                        try {
                            classes.add(Class.forName(className));
                        } catch (ClassNotFoundException e) {
                            throw new RuntimeException("ClassNotFoundException loading " + className);
                        }
                    }
                }
            } catch (IOException e) {
                throw new RuntimeException(pkgname + " (" + directory + ") does not appear to be a valid package", e);
            }
        }
        return classes;
    }


    private static class ImplementationPackage
    {
        private final static Logger _LOG = LoggerFactory.getLogger(ImplementationPackage.class);

        private final String _packageName;
        private final Class<? extends Device<?>> _deviceType;
        private final List<LoadableImplementation> _implementations;

        private ImplementationPackage(String packageName, Class<? extends Device<?>> deviceType, List<LoadableImplementation> implementations) {
            _packageName = packageName;
            _deviceType = deviceType;
            _implementations = implementations;
        }

        public void load(BackendContext context) {
            _LOG.debug("Loading implementations for package '" + _packageName + "' targeted at device type '" + _deviceType.getSimpleName() + "'...");
            for (LoadableImplementation implementation : _implementations) {
                try {
                    implementation.load(context);
                } catch (Exception e) {
                    _LOG.error("Failed to load implementation for operation '" + implementation.getOperation() + "' and algorithm '" + implementation.getAlgorithm() + "'!", e);
                }
            }
        }

    }

    private static class LoadableImplementation
    {
        private final Class<?> _implementationType;
        private final Class<? extends Operation> _targetOperation;
        private final Class<? extends Algorithm> _targetAlgorithm;
        private final Class<? extends Device<?>> _deviceType;

        private LoadableImplementation(
                Class<?> impl,
                Class<? extends Operation> targetOperation,
                Class<? extends DeviceAlgorithm<?>> algorithm,
                Class<? extends Device<?>> deviceType
        ) {
            _targetOperation = targetOperation;
            _targetAlgorithm = algorithm;
            _implementationType = impl;
            _deviceType = deviceType;
        }

        public Class<? extends Operation> getOperation() {
            return _targetOperation;
        }

        public Class<? extends Algorithm> getAlgorithm() {
            return _targetAlgorithm;
        }

        public void load(BackendContext context) {
            for ( Operation operation : context.getOperations() ) {
                if ( operation.getClass().equals(_targetOperation) ) {
                    for ( Algorithm algorithm : operation.getAllAlgorithms() ) {
                        if ( algorithm.getClass().equals(_targetAlgorithm) ) {
                            DeviceAlgorithm<?> deviceAlgorithm = (DeviceAlgorithm<?>) algorithm;
                            try {
                                _loadInternal(deviceAlgorithm);
                            } catch (Exception e) {
                                throw new RuntimeException("Failed to load implementation for operation '" + _targetOperation.getSimpleName() + "' and algorithm '" + _targetAlgorithm.getSimpleName() + "'!", e);
                            }
                        }
                    }
                }
            }
        }

        private void _loadInternal(DeviceAlgorithm<?> deviceAlgorithm) {
            // First we instantiate the implementation:
            ImplementationFor implementation = null;
            try {
                implementation = (ImplementationFor<?>) _implementationType.newInstance();
            } catch (InstantiationException | IllegalAccessException e) {
                throw new RuntimeException("Could not instantiate implementation class '"+_implementationType+"'!");
            }
            deviceAlgorithm.setImplementationFor(_deviceType, implementation);
        }
    }

}
