# 🤖 Grupo 19: Bertone - Ayala - Stoffel | IA 2026

Este repositorio contiene las entregas de la materia Inteligencia Artificial. Para asegurar que todos trabajemos con las mismas versiones de librerías, seguimos esta guía de configuración.

# 🚀 Guía de Configuración
Para correr el código y los tests sin errores, usamos `uv`, el gestor recomendado por la cátedra.

## 1. Requisitos Previos
Antes de empezar, asegúrense de tener instalado:
* **Python 3.14.5** (o superior).
* **GitHub Desktop** (para clonar y manejar el repo).
* **Graphviz**: Instalador manual para Windows [aquí](https://graphviz.org/download/#windows). 
  > **IMPORTANTE**: Durante la instalación, marquen la opción **"Add Graphviz to the system PATH"**. Si no lo hacen, los grafos no se verán.

## 2. Configuración del Entorno (Solo la primera vez)
Una vez clonado el repo, abran una terminal en la carpeta del proyecto y ejecuten:

### instalar `uv`
Si no lo tienen instalado, ejecuten en PowerShell:
```powershell
powershell -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"

Crear y Activar el Entorno Virtual
# Crear el entorno virtual
uv venv

# Activar el entorno (deberían ver "( .venv )" a la izquierda en la terminal)
.\.venv\Scripts\activate

# Instalar Dependencias del Proyecto
Usaremos el archivo requirements.txt para instalar todo de una vez:

uv pip install -r requirements.txt

# Actualizar Librerías
Si alguien instala una librería nueva (ej: uv pip install numpy), debe actualizar la lista oficial para el resto del grupo:

Ejecutar: uv pip freeze > requirements.txt
Subir el archivo requirements.txt a GitHub.

Los demás compañeros deberán ejecutar 
uv pip install -r requirements.txt para estar al día.