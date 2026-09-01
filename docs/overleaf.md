# Sincronización con Overleaf

El proyecto puede sincronizarse con una plantilla de Overleaf mediante su integración
Git. No se debe guardar el token en este repositorio, en `.env`, en el README ni dentro
de una URL remota.

## Preparación en Overleaf

1. Abrir el proyecto de la plantilla.
2. Entrar a **Integrations / Git** desde el menú del proyecto y copiar la URL Git.
3. Generar un Git authentication token desde Account Settings (o desde ese diálogo).

Overleaf solicita el usuario `git` y el token como contraseña. El token es una
credencial de acceso a todos los proyectos para los que tu cuenta tiene permisos y
debe mantenerse privado. Consulta la [documentación oficial de tokens Git](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/git/git-integration-authentication-tokens).

## Conectar este repositorio

Desde la raíz del proyecto, reemplazando `<GIT-URL>` por la URL copiada desde Overleaf:

```powershell
git remote add overleaf <GIT-URL>
git remote -v
```

Si la plantilla ya contiene archivos que se deben traer al repositorio local:

```powershell
git pull overleaf master --allow-unrelated-histories --rebase=false
```

Después de revisar y confirmar los cambios localmente:

```powershell
git add .
git commit -m "Sincroniza plantilla de investigación"
git push overleaf main:master
```

Cuando Git solicite credenciales, usar:

```text
Username: git
Password: <token de Overleaf>
```

El puente Git de Overleaf trabaja con una única historia lineal y una rama remota
`master`; por eso el último comando traduce la rama local `main` a `master`.

## Qué necesito para conectarlo aquí

Para dejar agregado el remote en este checkout necesito la URL Git del proyecto de
Overleaf. No necesito que pegues el token en el chat: puedes introducirlo de forma
interactiva en el primer `pull`/`push`, o usar un gestor de credenciales local. Si la
plantilla todavía no está sincronizada, también puedes descargarla como ZIP y colocar
los archivos `.tex`, `.bib` y recursos en una carpeta separada antes de decidir cómo
integrarlos con el código Python.

