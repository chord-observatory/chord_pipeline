"""Tasks for selecting subsets of the correlator inputs."""

from __future__ import annotations

import numpy as np
from caput import config, memdata
from caput.pipeline import tasklib

from ..core.container import INPUT_TYPE_DISH, INPUT_TYPE_RFI

# Groups in a container that do not hold datasets
_SKIP_GROUPS = {"index_map", "reverse_map", "history"}

_INPUT_TYPES = {"dish": [INPUT_TYPE_DISH], "rfi": [INPUT_TYPE_RFI], "all": None}


def _walk_datasets(group, root=""):
    """Get the names of all datasets in a container (including in groups)."""
    names = []
    for key, item in group.items():
        name = f"{root}{key}"
        if memdata.is_group(item):
            if name not in _SKIP_GROUPS:
                names += _walk_datasets(item, f"{name}/")
        else:
            names.append(name)
    return names


class SelectInputs(tasklib.base.ContainerTask):
    """Keep a subset of the inputs of an unstacked timestream.

    The inputs are selected by type (using the ``input_info/type`` dataset of
    a :class:`~chord_pipeline.core.container.CHORDTimeStream`) and/or by label.
    Only the products between two selected inputs are kept, and the products
    are re-indexed to the new input axis. Every dataset with an ``input``,
    ``prod`` or ``stack`` axis is sliced to match, all other datasets and the
    attributes are copied unchanged.

    Attributes
    ----------
    input_type : str
        Type of the inputs to keep: "dish", "rfi" or "all". Default is "dish".
    exclude_labels : list
        Labels of inputs to remove in addition (e.g. ``["A1p1", "A1p2"]``).
        Default is none.
    """

    input_type = config.enum(list(_INPUT_TYPES), default="dish")
    exclude_labels = config.Property(proptype=list, default=[])

    def process(self, data):
        """Select the inputs.

        Parameters
        ----------
        data : CHORDTimeStream
            Unstacked timestream to select from.

        Returns
        -------
        out : CHORDTimeStream
            Container of the same type holding only the selected inputs.
        """
        if data.is_stacked:
            raise ValueError("Can only select inputs from unstacked data.")

        data.redistribute("freq")

        # Find the inputs to keep
        keep = np.ones(len(data.input), dtype=bool)
        types = _INPUT_TYPES[self.input_type]
        if types is not None:
            keep &= np.isin(data["input_info/type"][:], types)

        labels = np.array(
            [
                x.decode() if isinstance(x, bytes) else str(x)
                for x in data.input["correlator_input"]
            ]
        )
        unknown = set(self.exclude_labels) - set(labels)
        if unknown:
            self.log.warning(f"Labels to exclude not found in data: {sorted(unknown)}")
        keep &= ~np.isin(labels, self.exclude_labels)

        isel = np.flatnonzero(keep)
        if len(isel) == 0:
            raise ValueError("No inputs selected.")

        # Find the products between kept inputs, and re-index them
        old_to_new = np.full(len(data.input), -1)
        old_to_new[isel] = np.arange(len(isel))

        prod = data.prod
        psel = np.flatnonzero(
            (old_to_new[prod["input_a"]] >= 0) & (old_to_new[prod["input_b"]] >= 0)
        )
        new_prod = np.empty(len(psel), dtype=prod.dtype)
        new_prod["input_a"] = old_to_new[prod["input_a"][psel]]
        new_prod["input_b"] = old_to_new[prod["input_b"][psel]]

        new_stack = np.empty(len(psel), dtype=data.stack.dtype)
        new_stack["prod"] = np.arange(len(psel))
        new_stack["conjugate"] = 0

        reverse_stack = np.empty(len(psel), dtype=[("stack", "<u4"), ("conjugate", "u1")])
        reverse_stack["stack"] = np.arange(len(psel))
        reverse_stack["conjugate"] = 0

        self.log.info(
            f"Keeping {len(isel)} of {len(data.input)} inputs and "
            f"{len(psel)} of {len(prod)} products."
        )

        out = data.__class__(
            input=data.input[isel],
            prod=new_prod,
            stack=new_stack,
            reverse_map_stack=reverse_stack,
            axes_from=data,
            attrs_from=data,
            skip_datasets=True,
            comm=data.comm,
            distributed=data.distributed,
        )

        # Copy the other axes (e.g. `xyz`, `file`) not defined by the container
        for axis, index_map in data.index_map.items():
            if axis not in out.index_map:
                out.create_index_map(axis, index_map)

        axis_sel = {"input": isel, "prod": psel, "stack": psel}

        for name in _walk_datasets(data._data):
            dset = data[name]
            axes = [
                a.decode() if isinstance(a, bytes) else str(a) for a in dset.attrs["axis"]
            ]
            sel = tuple(axis_sel.get(ax, slice(None)) for ax in axes)

            distributed = isinstance(dset, memdata.MemDatasetDistributed)
            src = dset[:].local_array if distributed else dset[:]

            # Index one axis at a time, as numpy can't combine several index arrays
            for ii, s in enumerate(sel):
                if not isinstance(s, slice):
                    src = np.take(src, s, axis=ii)

            if name in out.dataset_spec:
                new = out.add_dataset(name)
                if distributed:
                    new[:].local_array[:] = src
                else:
                    new[:] = src
            else:
                new = out.create_dataset(
                    name,
                    data=np.ascontiguousarray(src),
                    distributed=False,
                )
                new.attrs["axis"] = dset.attrs["axis"]

            # Copy any other dataset attributes
            for key, val in dset.attrs.items():
                if key != "axis":
                    new.attrs[key] = val

        out.attrs["num_elements"] = len(isel)
        out.attrs["num_prod"] = len(psel)

        return out


class RemoveRFIMonitors(SelectInputs):
    """Remove the RFI monitor inputs, keeping only the dish inputs.

    Equivalent to :class:`SelectInputs` with ``input_type: dish``. This
    reduces the 48 input (1176 product) pathfinder data to 32 inputs (528
    products).
    """

    input_type = config.enum(["dish"], default="dish")
