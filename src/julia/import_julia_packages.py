import json
import os
import juliapkg
from juliacall import Main as jl
from juliacall import Pkg as jl_pkg

# Ensure Julia and packages are installed
juliapkg.resolve()

# List of packages to include in the JSON
packages_to_include = [
    "DifferentialEquations",
    "MethodOfLines",
    "DomainSets",
    "ModelingToolkit",
    "OrdinaryDiffEq",
    "Interpolations",
    "PyCall"

]

# Check if the specified packages are installed, and add them if not
installed_packages = jl_pkg.installed()
for package in packages_to_include:
    if package not in installed_packages:
        print(f"{package}.jl is not installed. Adding it now...")
        jl_pkg.add(package)
    else:
        print(f"{package}.jl is already installed.")

# Resolve after checking and potentially adding the packages
juliapkg.resolve()

# Get dependencies
dependencies = jl_pkg.dependencies()

print(f"Julia Version: {jl.VERSION}\n")

# Initialize the structure for the JSON file
juliapkg_json = {
    "julia": "~1.6.7, ~1.7, ~1.8, ~1.9, =1.10.0, ~1.10.4",
    "packages": {}
}

# Filter dependencies and construct the JSON structure
for uuid, PackageInfo in dependencies.items():
    if PackageInfo.name in packages_to_include:
        juliapkg_json["packages"][PackageInfo.name] = {
            "uuid": str(uuid),
            "version": f"={str(PackageInfo.version)}"
        }

# Write the JSON structure to a file
json_file_path = 'src/juliapkg.json'
with open(json_file_path, 'w') as json_file:
    json.dump(juliapkg_json, json_file, indent=4)

# Verify the JSON file content
with open(json_file_path, 'r') as json_file:
    try:
        data = json.load(json_file)
        print("JSON file content is valid.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        # Clean up the invalid JSON file
        os.remove(json_file_path)
        print(f"Invalid JSON file {json_file_path} has been removed.")

# Print the installed packages for verification
# print(f"Inspect installed packages: {jl_pkg.installed()}")


# print(f"Inspect dependencies: {jl_pkg.dependencies()}")

# print(f"Inspect installed packages: {jl_pkg.installed()}")

# dependencies = jl_pkg.dependencies()

# for i, (uuid, PackageInfo) in enumerate(dependencies.items()):
#     if i >= 1:
#         break
#     print(f"UUID: {uuid}\n")
#     print(f"PackageInfo: {PackageInfo}\n")
#     print(f"PackageInfo.name: {PackageInfo.name}\n")
#     print(f"PackageInfo.version: {PackageInfo.version}\n")
#     print(f"PackageInfo.source: {PackageInfo.source}\n")

